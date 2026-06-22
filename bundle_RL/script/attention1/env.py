import copy

import gymnasium as gym
import gurobipy as gp
import numpy as np
import torch

from bundle_RL.script.lag_problem import SubProblem

"""
Attention1 Bundle Environment（简化版）

state：当前的所有cuts，valid_mask，当前的pi值，当前的trial_point、以及场景 realization
action：lambda 和 步长（policy 输出的 raw 值）

状态转移：
    raw_lambda → masked_fill(valid_mask==0, -inf) → softmax → lambdas
    raw_eta    → sigmoid → eta
    d_t = Σ λᵢ gᵢ
    π_new = π + η · d_t
    SubProblem.solve(π_new) → g_new, φ_new

reward：基于子问题目标函数的提升值 (φ_new - φ_prev) / scale

特点:
1. 使用 valid_mask 标记有效的 cuts
2. mask + softmax 顺序：先 mask_fill(-inf)，再 softmax（数值稳定，分布严格在有效集合上）
3. bundle 数据结构预留 age 和 error 字段
"""


class BundleDualEnv(gym.Env):
    def __init__(self, logger, config, n, state_dim, K, verbose=False, tolerance=1e-3):
        """
        :param logger: 日志器
        :param config: BundleConfig 对象
        :param n: realization 索引
        :param state_dim: cut中次梯度的维度
        :param K: 使用padding的方式，K代表最大的cuts数，同时也是输出的lambda维度
        :param verbose: 是否输出详细日志（用于test模式）
        """
        super().__init__()

        self.subproblem = SubProblem(logger, config, n)
        self.K = K
        self.state_dim = state_dim
        self.action_dim = K + 1  # 输出lambda以及步长
        self.logger = logger
        self.verbose = verbose

        # Master problem 参数 (仅训练时使用)
        self.tolerance = tolerance
        self.m_l = 0.2
        self.m_r = 0.5
        self.u_min = 0.1
        self.training = not verbose  # verbose=True 表示推理/测试模式

        self.training = True

        # 保存 config 用于获取额外特征
        self.config = config
        self.n = n  # realization 索引

        # 从 config 中获取 trial_point 并展平
        self.trial_point = self._flatten_trial_point(config.trial_point)
        self.trial_point_dim = len(self.trial_point)

        # 从 PROBLEM_PARAMS 中获取当前阶段和realization的数据
        self.problem_params = config.PROBLEM_PARAMS
        self.stage = config.T

        # 获取当前 realization 的数据
        self.p_d = np.array(self.problem_params.p_d[self.stage][self.n], dtype=np.float32)
        self.re = np.array(self.problem_params.re[self.stage][self.n], dtype=np.float32)

        # 计算 realization 特征维度（不包含 prob）
        self.realization_dim = len(self.p_d) + len(self.re)

        # ========== 状态空间 ==========
        self.observation_space = gym.spaces.Dict({
            "cuts": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.K, self.state_dim),
                dtype=np.float32
            ),
            "valid_mask": gym.spaces.Box(
                low=0,
                high=1,
                shape=(self.K,),
                dtype=np.float32
            ),
            "pi": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.state_dim,),
                dtype=np.float32
            ),
            "trial_point": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.trial_point_dim,),
                dtype=np.float32
            ),
            "realization": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.realization_dim,),
                dtype=np.float32
            )
        })

        # ========== 动作空间 ==========
        # 前K维是 raw_lambda（将在 step 中经 softmax 归一化），最后1维是 raw_eta（经 sigmoid）
        # 使用宽范围 [-10, 10]：
        #   - softmax 在 [-10,10] 范围内可产生从均匀到接近 one-hot 的全部分布
        #   - sigmoid(-10)≈0, sigmoid(10)≈1，步长 eta 覆盖 (0,1) 全域
        #   - 避免窄 action_space 导致 SB3 clip 破坏策略梯度信号
        self.action_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.action_dim,),
            dtype=np.float32
        )

        self.reset()

    def _flatten_trial_point(self, trial_point):
        """
        将 trial_point 展平为一维数组
        trial_point = (X_TRIAL, Y_TRIAL, X_BS_TRIAL, SOC_TRIAL)
        """
        X_TRIAL, Y_TRIAL, X_BS_TRIAL, SOC_TRIAL = trial_point

        # 展平 X_BS_TRIAL（二维列表）
        X_BS_flat = [val for bs in X_BS_TRIAL for val in bs]

        # 合并所有部分
        flat = np.array(X_TRIAL + Y_TRIAL + X_BS_flat + SOC_TRIAL, dtype=np.float32)
        return flat

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.bundle = []
        self.pi = np.zeros(self.state_dim)  # 初始化pi
        self.t = 0  # 迭代次数

        self.best_phi = 0

        # 初始solve
        g, phi = self.subproblem.solve(self.pi)

        # 使用第一次计算的g对reward进行缩放
        self.scale = np.linalg.norm(g) + 1e-8
        self.phi_prev = phi

        sub_result = {
            "pi": self.pi.copy(),
            "g": g.copy(),
            "phi": phi,
            "age": 0,
            "error": 0.0
        }

        self.bundle.append(sub_result)

        # 初始化 Master Problem (仅训练时)
        if self.training:
            self._init_master()
            self._add_cut(self.pi, phi, g)
            self.master_x_best = self.pi.copy()
            self.master_f_best = phi
            self.master_u = 1.0
            self.master_i_u = 0
            self.master_var_est = 1e9
            self.ub = None

        return self._get_state(), {}

    # ---------- Master Problem 方法 (仅训练时使用) ----------

    def _compute_diversity_reward(self, g_new):
        """
        计算多样性奖励 r_div = 1 - max_i cos(g_i, g_new)
        衡量新子梯度与 bundle 中已有子梯度的最大余弦相似度，
        相似度越低说明新 cut 提供的信息越多样，奖励越高。
        """
        g_norm = np.linalg.norm(g_new)
        if g_norm < 1e-12:
            return 0.0

        max_sim = -1.0
        for cut in self.bundle:
            g_i = cut["g"]
            g_i_norm = np.linalg.norm(g_i)
            if g_i_norm < 1e-12:
                continue
            sim = np.dot(g_i, g_new) / (g_i_norm * g_norm)
            max_sim = max(max_sim, sim)

        return 1.0 - max_sim

    def _init_master(self):
        """初始化 Master Problem 的 Gurobi 模型"""
        self.master_model = gp.Model("Master_Bundle")
        self.master_model.setParam("OutputFlag", 0)
        self.master_v = self.master_model.addVar(lb=-gp.GRB.INFINITY, name="v")
        self.master_x_vars = self.master_model.addVars(self.state_dim, lb=-gp.GRB.INFINITY, name="x")
        self.master_cuts_constraints = []
        self.master_iter_idx = 0

    def _add_cut(self, x_new, f_new, g_new):
        """向 Master Problem 添加一个 cut"""
        self.master_iter_idx += 1
        cut_expr = f_new + gp.quicksum(
            g_new[j] * (self.master_x_vars[j] - x_new[j]) for j in range(self.state_dim)
        )
        constr = self.master_model.addConstr(self.master_v <= cut_expr, name=f"cut_{self.master_iter_idx}")
        self.master_cuts_constraints.append(constr)

    def _solve_master(self):
        """求解 Master Problem，返回 (ub, x_candidate)，求解失败返回 (None, None)"""
        u = self.master_u
        obj = self.master_v - u / 2 * gp.quicksum(
            (self.master_x_vars[j] - self.master_x_best[j]) ** 2 for j in range(self.state_dim)
        )
        self.master_model.setObjective(obj, gp.GRB.MAXIMIZE)
        self.master_model.optimize()

        if self.master_model.status != gp.GRB.OPTIMAL:
            return None, None

        x_candidate = np.array([self.master_x_vars[j].x for j in range(self.state_dim)])
        ub = self.master_v.x
        return ub, x_candidate

    def _update_strategy(self, x_new, f_new, g_new, ub):
        """Weight update 逻辑 (移植自 lag_problem.py)"""
        if self.master_iter_idx <= 1:
            return

        delta = ub - self.master_f_best
        rel_gap = delta / max(abs(self.master_f_best), 1)

        serious_step = (f_new - self.master_f_best) >= self.m_l * rel_gap

        u_int = 2 * self.master_u * (1 - (f_new - self.master_f_best) / delta) if abs(delta) > 1e-12 else self.master_u
        u = self.master_u

        if serious_step:
            weight_too_large = (f_new - self.master_f_best) >= (self.m_r * delta)
            if weight_too_large and self.master_i_u > 0:
                u = u_int
            elif self.master_i_u > 3:
                u = self.master_u / 2
            u_new = max(u, self.master_u / 10, self.u_min)
            self.master_var_est = max(self.master_var_est, 2 * delta)
            self.master_i_u = max(self.master_i_u + 1, 1) if u_new == self.master_u else 1
        else:
            p = -self.master_u * (np.array(x_new) - np.array(self.master_x_best))
            alpha = delta - np.linalg.norm(p, ord=2) ** 2 / self.master_u
            self.master_var_est = min(self.master_var_est, np.linalg.norm(p, ord=1) + alpha)
            linearization_error = f_new + np.dot(g_new, self.master_x_best - x_new) - self.master_f_best
            if linearization_error > max(self.master_var_est, 10 * delta) and self.master_i_u < -3:
                u = u_int
            u_new = min(u, 10 * self.master_u)
            self.master_i_u = min(self.master_i_u - 1, -1) if u_new == self.master_u else -1

        self.master_u = u_new

        if serious_step:
            self.master_x_best = x_new.copy()
            self.master_f_best = f_new

    # --------------------------------------------------

    def step(self, action):
        """
        action = [raw_lambda_1 ... raw_lambda_K , raw_eta]

        简化版处理流程：
            1. 对 raw_lambda 用 valid_mask 做 mask_fill(-inf)
            2. softmax 得到合法分布 lambdas
            3. raw_eta 经 sigmoid 得到 eta ∈ (0, 1)
            4. d_t = lambdas @ G,  pi_new = pi + eta * d_t
        """
        # 拆分动作
        raw_lambda = np.asarray(action[:self.K], dtype=np.float32).copy()
        raw_eta = float(action[-1])

        # ---------- 取 state ----------
        state = self._get_state()
        G = state["cuts"]
        valid_mask = state["valid_mask"]

        # ---------- lambda：mask + softmax ----------
        # 先 mask 无效位置（设为 -inf），再做 softmax
        masked_logits = np.where(valid_mask > 0, raw_lambda, -np.inf)
        # 数值稳定的 softmax
        max_logit = np.max(masked_logits[np.isfinite(masked_logits)]) if np.any(np.isfinite(masked_logits)) else 0.0
        exp_lambda = np.exp(masked_logits - max_logit)
        exp_lambda = np.where(np.isfinite(exp_lambda), exp_lambda, 0.0)
        denom = np.sum(exp_lambda) + 1e-8
        lambdas = exp_lambda / denom

        # ---------- 步长映射 ----------
        # sigmoid 保证 eta ∈ (0, 1)
        eta = 1.0 / (1.0 + np.exp(-raw_eta))
        # eta = 0.5
        # eta = np.clip(eta, 0.1, 0.9)



        # ---------- 方向构造与 pi 更新 ----------
        d = lambdas @ G  # (state_dim,)
        print("############bundle_RL#########")
        print("lambda = ", lambdas)
        print("eta = ", eta)
        print("d_norm = ", np.linalg.norm(d, axis=0))

        self.pi = self.pi + eta * d

        # 子问题求解
        g, phi_new = self.subproblem.solve(self.pi)

        # ========== Master 计算 (仅训练时) ==========
        phi_master = None
        if self.training:
            self._add_cut(self.pi, phi_new, g)
            self.ub, pi_master = self._solve_master()
            if self.ub is not None:
                g_master, phi_master = self.subproblem.solve(pi_master)
                self._update_strategy(self.pi, phi_new, g, self.ub)
            else:
                self.ub = None

        # reward 使用子问题的目标函数的提升值
        # reward = (phi_new - self.bundle[-1]["phi"]) / self.scale

        best_phi_old = self.best_phi
        self.best_phi = max(
            self.best_phi,
            phi_new
        )

        # # 判断是否是第一次取得有效进展（即从 0 突破）
        # if best_phi_old < 1e-6:
        #     if phi_new > 0:
        #         reward = 0.5  # 或者设为 0.5，给一个温和的初始启动奖励
        #     else:
        #         reward = 0.0
        # else:
        #     # 后面恢复正常的百分比提升奖励
        #     reward = (phi_new - best_phi_old) / best_phi_old


        # reward = (phi_new - best_phi_old) / self.scale
        # reward = -np.log1p(max(0, -improve))
        # reward = np.sign(improve) * np.log1p(abs(improve))
        # reward -= 0.01

        # 稀疏的奖励设置
        # best_phi_old = self.best_phi
        # best_phi_new = max(
        #     self.best_phi,
        #     phi_new
        # )
        # reward = (best_phi_new - best_phi_old) / self.scale
        # self.best_phi = best_phi_new

        # 多样性奖励: 鼓励探索与已有 cut 不同的方向
        # r_div = self._compute_diversity_reward(g)
        # reward += r_div

        # print("reward: ", reward)

        reward = 0


        # 附加 reward 项: RL 策略的 phi - master 策略的 phi (仅训练时)
        # if self.training and phi_master is not None:
        #     reward += (phi_new - phi_master) / self.scale


        # 更新 cut age
        for cut in self.bundle:
            cut["age"] += 1

        cut_new = {
            "pi": self.pi.copy(),
            "g": g.copy(),
            "phi": phi_new,
            "age": 0,
            "error": 0.0
        }

        self.bundle.append(cut_new)

        self.t += 1
        terminated = self.t >= self.K

        if terminated:
            rel_gap = (
                              self.ub - self.best_phi
                      ) / max(abs(self.best_phi), 1)
            rel_gap = max(rel_gap, 1e-8)
            reward = -np.log(
                rel_gap
            )

        # Gap 停止条件 (仅训练时)
        # if self.training and self.ub is not None and not terminated:
        #     rel_gap = (self.ub - self.best_phi) / max(abs(self.best_phi), 1)
        #     if rel_gap <= self.tolerance:
        #         reward += 5
        #         terminated = True

        # 记录每次step的输出值（仅在verbose模式下）
        if self.verbose:
            self.logger.debug(f"[BundleEnv Step {self.t}] "
                              f"raw_eta={raw_eta:.4f}, "
                              f"eta={eta:.4f}, "
                              f"pi_norm={np.linalg.norm(self.pi):.6f}, "
                              f"phi_new={phi_new:.6f}, "
                              f"reward={reward:.6f}, "
                              f"active_cuts={int(np.sum(valid_mask))}, "
                              f"terminated={terminated}")

        if not np.isfinite(reward):
            print("reward nan")
            print("ub =", self.ub)
            print("best_phi =", self.best_phi)
            print("rel_gap =", rel_gap)
            raise ValueError

        return self._get_state(), reward, terminated, False, {}

    # --------------------------------------------------
    def _get_state(self):
        """
        获取当前最新的状态，从self.bundle中抽取最新的数据，padding出cuts矩阵
        :return: cuts，valid_mask，pi，trial_point，realization
        """
        cuts = np.zeros((self.K, self.state_dim), dtype=np.float32)
        valid_mask = np.zeros(self.K, dtype=np.float32)

        # 取出最后K个最新数据（为了应对迭代次数超过K的情况，丢弃旧数据）
        active = self.bundle[-self.K:]

        start = self.K - len(active)

        for i, cut in enumerate(active):
            cuts[start + i] = cut["g"]
            valid_mask[start + i] = 1.0

        # 构建 realization 特征向量（不包含 prob）
        realization_feature = np.concatenate([
            self.p_d,
            self.re
        ])

        return {
            "cuts": cuts,
            "valid_mask": valid_mask,
            "pi": self.pi.astype(np.float32),
            "trial_point": self.trial_point,
            "realization": realization_feature
        }

    @classmethod
    def create_env(cls, logger, config, tolerance=1e-5, verbose=False, K=20):
        """创建单个环境（使用 config 中的 n 参数）"""
        env = cls(
            logger=logger,
            config=config,
            n=config.n,
            state_dim=config.N_VARS,
            K=K,
            verbose=verbose,
            tolerance=tolerance
        )
        return env, None
