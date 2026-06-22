import copy

import gymnasium as gym
import gurobipy as gp
import numpy as np
from bundle_RL.script.lag_problem import SubProblem

"""
尝试增加输入的feature

state：当前的所有cuts，当前的pi值，当前的trail_point、以及场景 realization
action：lambda 和 步长

状态转移：lambda + 步长 -> 归一化 -> pi -> sub求解得到子问题

reward：pi对应的子问题最优解对应的目标函数值，求解的真实值，让子问题的解尽可能大

"""


class BundleDualEnv(gym.Env):
    def __init__(self, logger, config, n, state_dim, K, verbose=False, tolerance=1e-5):
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
        
        # 保存 config 用于获取额外特征
        self.config = config
        self.n = n  # realization 索引
        
        # 从 config 中获取 trial_point 并展平
        self.trial_point = self._flatten_trial_point(config.trial_point)
        self.trial_point_dim = len(self.trial_point)
        
        # 从 PROBLEM_PARAMS 中获取当前阶段和realization的数据
        # config.T 是当前阶段索引（0-based），config.n 是 realization 索引
        self.problem_params = config.PROBLEM_PARAMS
        self.stage = config.T
        
        # 获取当前 realization 的数据
        self.p_d = np.array(self.problem_params.p_d[self.stage][self.n], dtype=np.float32)
        self.re = np.array(self.problem_params.re[self.stage][self.n], dtype=np.float32)
        self.prob = float(self.problem_params.prob[self.stage][self.n])

        # 计算 realization 特征维度（不包含 prob）
        self.realization_dim = len(self.p_d) + len(self.re) + 1 # p_d + re

        # shape = (K, state_dim)
        # 使用Box，padding部分为0
        self.observation_space = gym.spaces.Dict({
            "cuts": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.K, self.state_dim),
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
        # 前K维是lambda，最后1维是步长
        self.action_space = gym.spaces.Box(
            low=-10,
            high=10,
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
        self.pi = np.zeros(self.state_dim)
        self.t = 0  # 迭代次数

        self.best_phi = 0

        # 初始solve
        g, phi = self.subproblem.solve(self.pi)

        # 使用第一次计算的g对reward进行缩放
        self.scale = np.linalg.norm(g) + 1e-8

        sub_result = {
            "pi": self.pi.copy(),
            "g": g.copy(),
            "phi": phi,
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

    # ---------- Master Problem 方法 (仅训练时使用) ----------

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
        """求解 Master Problem，返回 (ub, x_candidate)"""
        u = self.master_u
        obj = self.master_v - u / 2 * gp.quicksum(
            (self.master_x_vars[j] - self.master_x_best[j]) ** 2 for j in range(self.state_dim)
        )
        self.master_model.setObjective(obj, gp.GRB.MAXIMIZE)
        self.master_model.optimize()

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
        action = [lambda_1 ... lambda_K , eta]
        """
        # 拆分动作
        raw_lambda = action[:self.K]
        raw_eta = action[-1]

        # ---------- lambda 归一化 ----------
        temperature = 0.1
        raw_lambda = raw_lambda / temperature
        exp_lambda = np.exp(raw_lambda)
        lambdas = exp_lambda / (np.sum(exp_lambda) + 1e-8)

        # 查看lambda的熵，分布的尖锐程度
        entropy = -np.sum(
            lambdas * np.log(lambdas + 1e-8)
        )
        max_lambda = np.max(lambdas)
        print("entropy", entropy)
        print("max_lambda", max_lambda)

        # ---------- 步长映射 ----------
        # 用sigmoid保证正值，并限制最大步长
        # TODO: 步长的上界具体设置可以查看bundle算法中的步长大小
        eta = 1.0 * (1 / (1 + np.exp(-raw_eta)))
        # eta = np.clip(eta, 0.1, 0.9)
        # eta = 0.5

        # ---------- 用 state 聚合 ----------
        state = self._get_state()
        G = state["cuts"]
        d = lambdas @ G  # (state_dim,)

        print("############ bundle_RL #########")
        print("lambda = ", lambdas)
        print("eta = ", eta)

        # 更新pi
        self.pi = self.pi + eta * d

        # 子问题求解
        g, phi_new = self.subproblem.solve(self.pi)

        # ========== Master 计算 (仅训练时) ==========
        if self.training:
            self._add_cut(self.pi, phi_new, g)
            self.ub, pi_master = self._solve_master()
            lambda_master = [cut.Pi for cut in self.master_cuts_constraints]
            g_master, phi_master = self.subproblem.solve(pi_master)
            self._update_strategy(self.pi, phi_new, g, self.ub)

        cut_new = {
            "pi": self.pi.copy(),
            "g": g.copy(),
            "phi": phi_new,
        }

        # reward 使用子问题的目标函数的提升值
        reward = (phi_new - self.bundle[-1]["phi"]) / self.scale
        reward -= 0.1
        # best_phi_old = self.best_phi
        # self.best_phi = max(
        #     self.best_phi,
        #     phi_new
        # )
        # improve = phi_new - best_phi_old
        # reward = np.sign(improve) * np.log1p(abs(improve))


        # 稀疏的奖励设置
        # best_phi_old = self.best_phi
        # best_phi_new = max(
        #     self.best_phi,
        #     phi_new
        # )
        # reward = (best_phi_new - best_phi_old) / self.scale
        # self.best_phi = best_phi_new

        # 附加 reward 项: RL 策略的 phi - master 策略的 phi (仅训练时)
        # if self.training and phi_master is not None:
        #     reward += (phi_new - phi_master) / self.scale


        # 多样性cut
        # r_div = self._compute_diversity_reward(g)
        # reward += r_div


        self.bundle.append(cut_new)

        self.t += 1
        terminated = self.t >= self.K

        # Gap 停止条件 (仅训练时)
        if self.training and self.ub is not None and not terminated:
            rel_gap = (self.ub - self.best_phi) / max(abs(self.best_phi), 1)
            if rel_gap <= self.tolerance:
                terminated = True
                reward += 5
        
        # 记录每次step的输出值（仅在verbose模式下）
        if self.verbose:
            self.logger.debug(f"[BundleEnv Step {self.t}] "
                             f"raw_lambda={raw_lambda},"
                             # f"raw_lambda_sum={np.sum(raw_lambda):.4f}, "
                             f"raw_eta={raw_eta:.4f}, "
                             f"eta={eta:.4f}, "
                             f"pi_norm={np.linalg.norm(self.pi):.6f}, "
                             f"phi_new={phi_new:.6f}, "
                             f"reward={reward:.6f}, "
                             f"terminated={terminated}")

        assert np.isfinite(reward), f"reward={reward}"

        return self._get_state(), reward, terminated, False, {}

    # --------------------------------------------------
    def _get_state(self):
        """
        获取当前最新的状态，从self.bundle中抽取最新的数据，padding出cuts矩阵
        :return: cuts，pi，trial_point，realization
        """
        cuts = np.zeros((self.K, self.state_dim), dtype=np.float32)
        # 取出最后K个最新数据（为了应对迭代次数超过K的情况，丢弃旧数据）
        active = self.bundle[-self.K:]

        start = self.K - len(active)

        for i, cut in enumerate(active):
            cuts[start + i] = cut["g"]

        # 构建 realization 特征向量（不包含 prob）
        realization_feature = np.concatenate([
            self.p_d,
            self.re,
            np.array([self.prob], dtype=np.float32)
        ])

        return {
            "cuts": cuts,
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
