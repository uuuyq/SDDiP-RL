import gymnasium as gym
import numpy as np
import gurobipy as gp
import copy
from bundle_RL.script.lag_problem import SubProblem

"""
RL 预测步长 eta 的环境

算法流程：
1. RL 预测 eta
2. 将 eta 作为参数送入 master 问题求解（u = 2*eta）
3. 得到 d 以及新的 pi 值
4. 求解子问题得到 cut
5. 更新 state，进入下一个阶段

Observation:
- log_eta: log(eta)
- log_delta: log(delta)
- serious_step: 01值，上一步是否成功改善 center
- d_norm_sq: ||d||²
- proximal_term: eta*||d||²
- lin_error: linearization error
- gap_improve_1/2/3: gap 的改善
- g_norm_sq: ||g_t||²
- cos_gd: cos(g_t, d)

Action:
- a ∈ [-1, 1]
- η_new = η_old * exp(0.5*a)
"""


class BundleDualEnv(gym.Env):
    def __init__(self, logger, config, n, state_dim, K, verbose=False):
        """
        :param logger: 日志器
        :param config: BundleConfig 对象
        :param n: realization 索引
        :param state_dim: cut中次梯度的维度
        :param K: 最大迭代次数
        :param verbose: 是否输出详细日志
        """
        super().__init__()

        self.subproblem = SubProblem(logger, config, n)
        self.K = K
        self.state_dim = state_dim
        self.action_dim = 1  # 只输出一个标量动作
        self.logger = logger
        self.verbose = verbose
        
        # 保存 config 用于获取额外特征
        self.config = config
        self.n = n
        
        # 从 config 中获取 trial_point 并展平
        self.trial_point = self._flatten_trial_point(config.trial_point)
        self.trial_point_dim = len(self.trial_point)
        
        # 从 PROBLEM_PARAMS 中获取当前阶段和 realization 的数据
        self.problem_params = config.PROBLEM_PARAMS
        self.stage = config.T
        
        # 获取当前 realization 的数据
        self.p_d = np.array(self.problem_params.p_d[self.stage][self.n], dtype=np.float32)
        self.re = np.array(self.problem_params.re[self.stage][self.n], dtype=np.float32)
        self.prob = float(self.problem_params.prob[self.stage][self.n])

        # Master problem 参数
        self.m_l = 0.2
        self.m_r = 0.5
        self.u_min = 0.1

        # ========== Observation Space ==========
        self.observation_space = gym.spaces.Dict({
            "log_eta": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "log_delta": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "serious_step": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "d_norm_sq": gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "proximal_term": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "lin_error": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "gap_improve_1": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "gap_improve_2": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "gap_improve_3": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "g_norm_sq": gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "cos_gd": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        })

        # ========== Action Space ==========
        # a ∈ [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
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

        # ========== 初始化 Master Problem ==========
        self._init_master()

        # ========== 初始化变量 ==========
        self.pi = np.zeros(self.state_dim)  # 当前的 pi
        self.x_best = np.zeros(self.state_dim)  # 最优的 pi（稳定中心）
        self.x_new = np.zeros(self.state_dim)  # 最新求解的 pi
        self.t = 0  # 迭代次数
        
        # 状态变量
        self.eta = 0.5  # 初始 eta
        self.g_new = None  # 最新的子梯度
        self.f_new = None  # 最新的子问题目标值
        self.f_best = None  # 最优的子问题目标值
        self.d = None  # 最新的搜索方向
        self.delta = None  # 最新的 delta
        self.serious_step = False  # 上一步是否是 serious step
        self.i_u = 0  # weight update 计数器
        self.gap_history = []  # 历史 gap 记录
        self.lin_error = 0.0  # linearization error
        self.cuts_storage = []  # 存储所有 cuts

        # ========== 初始求解 ==========
        self.g_new, self.f_new = self.subproblem.solve(self.pi)
        self.f_best = self.f_new
        self.x_best = self.pi.copy()
        self.cuts_storage.append((self.g_new.copy(), self.pi.copy(), self.f_new))
        self._add_cut(self.pi, self.f_new, self.g_new)
        
        # 用第一次的梯度缩放 reward
        self.scale = np.linalg.norm(self.g_new) + 1e-8

        return self._get_state(), {}

    def _init_master(self):
        """初始化 Gurobi 模型"""
        self.model = gp.Model("Master_Bundle")
        self.model.setParam("OutputFlag", 0)
        self.v = self.model.addVar(lb=-gp.GRB.INFINITY, name="v")
        self.x_vars = self.model.addVars(self.state_dim, lb=-gp.GRB.INFINITY, name="x")
        self.cuts_constraints = []
        self.iter_idx = 0

    def _add_cut(self, x_new, f_new, g_new):
        """向 Master Problem 添加一个 cut"""
        self.iter_idx += 1
        cut_expr = f_new + gp.quicksum(
            g_new[j] * (self.x_vars[j] - x_new[j]) for j in range(self.state_dim)
        )
        constr = self.model.addConstr(self.v <= cut_expr, name=f"cut_{self.iter_idx}")
        self.cuts_constraints.append(constr)

    def _solve_master(self, eta):
        """
        求解 Master Problem
        :param eta: RL 预测的 eta，u = 2 * eta
        :return: ub（上界）, x_candidate（新的 pi）
        """
        u = 2 * eta
        
        # 设置目标函数: obj = v - u/2 * ||x - x_best||^2
        obj = self.v - u / 2 * gp.quicksum(
            (self.x_vars[j] - self.x_best[j]) ** 2 for j in range(self.state_dim)
        )
        self.model.setObjective(obj, gp.GRB.MAXIMIZE)
        self.model.optimize()

        x_candidate = np.array([self.x_vars[j].x for j in range(self.state_dim)])
        ub = self.v.x

        return ub, x_candidate

    # --------------------------------------------------

    def step(self, action):
        """
        action = [a], a ∈ [-1, 1]
        """
        # ========== 1. 更新 eta ==========
        a = action[0]
        eta_old = self.eta
        self.eta = eta_old * np.exp(0.5 * a)

        # ========== 2. 求解 Master Problem ==========
        ub, self.x_new = self._solve_master(self.eta)
        self.d = self.x_new - self.pi

        # ========== 3. 求解 Sub Problem ==========
        self.g_new, self.f_new = self.subproblem.solve(self.x_new)

        # ========== 4. 更新状态 ==========
        # 计算 delta
        if self.f_best is None:
            self.delta = 1.0
        else:
            self.delta = (ub - self.f_best) / max(abs(self.f_best), 1)
        
        # 判断 serious_step（使用与 lag_problem 相同的逻辑）
        self.serious_step = (self.f_new - self.f_best) >= self.m_l * self.delta
        
        # 计算 linearization error
        if self.f_best is not None and self.g_new is not None:
            self.lin_error = (self.f_new + np.dot(self.g_new, self.x_best - self.x_new) - self.f_best)
        
        # 更新 gap history
        self.gap_history.append(self.delta)
        
        # 更新 x_best 和 f_best（基于稳定中心）
        old_f_best = self.f_best
        if self.serious_step:
            self.x_best = self.x_new.copy()
            self.f_best = self.f_new

        # ========== 5. 添加 cut ==========
        self._add_cut(self.x_new, self.f_new, self.g_new)
        self.cuts_storage.append((self.g_new.copy(), self.x_new.copy(), self.f_new))

        # ========== 6. 计算 Reward ==========
        # 使用基于稳定中心的提升
        if old_f_best is not None:
            reward = (self.f_best - old_f_best) / self.scale
        else:
            reward = 0.0

        # ========== 7. 结束条件 ==========
        self.t += 1
        terminated = self.t >= self.K

        # 更新 pi
        self.pi = self.x_new.copy()

        # ========== 8. 记录日志 ==========
        if self.verbose:
            self.logger.debug(f"[BundleEnv Step {self.t}] "
                             f"eta_old={eta_old:.4f}, "
                             f"action={a:.4f}, "
                             f"eta_new={self.eta:.4f}, "
                             f"delta={self.delta:.6e}, "
                             f"serious_step={self.serious_step}, "
                             f"f_new={self.f_new:.6f}, "
                             f"f_best={self.f_best:.6f}, "
                             f"pi_norm={np.linalg.norm(self.pi):.6f}, "
                             f"reward={reward:.6f}, "
                             f"terminated={terminated}")

        return self._get_state(), reward, terminated, False, {}

    # --------------------------------------------------
    def _get_state(self):
        """
        构建 observation 字典
        """
        # 处理初始状态
        if self.d is None or self.delta is None:
            return {
                "log_eta": np.array([np.log(self.eta)], dtype=np.float32),
                "log_delta": np.array([0.0], dtype=np.float32),
                "serious_step": np.array([0.0], dtype=np.float32),
                "d_norm_sq": np.array([0.0], dtype=np.float32),
                "proximal_term": np.array([0.0], dtype=np.float32),
                "lin_error": np.array([0.0], dtype=np.float32),
                "gap_improve_1": np.array([0.0], dtype=np.float32),
                "gap_improve_2": np.array([0.0], dtype=np.float32),
                "gap_improve_3": np.array([0.0], dtype=np.float32),
                "g_norm_sq": np.array([np.linalg.norm(self.g_new) ** 2], dtype=np.float32),
                "cos_gd": np.array([0.0], dtype=np.float32)
            }

        # 计算 gap_improve_1/2/3
        n_hist = len(self.gap_history)
        gap_improve_1 = 0.0
        gap_improve_2 = 0.0
        gap_improve_3 = 0.0

        current_log_delta = np.log(np.clip(self.delta, 1e-10, 1e10))
        if n_hist >= 2:
            gap_improve_1 = current_log_delta - np.log(np.clip(self.gap_history[-2], 1e-10, 1e10))
        if n_hist >= 3:
            gap_improve_2 = current_log_delta - np.log(np.clip(self.gap_history[-3], 1e-10, 1e10))
        if n_hist >= 4:
            gap_improve_3 = current_log_delta - np.log(np.clip(self.gap_history[-4], 1e-10, 1e10))

        # 计算 cos(g_t, d)
        cos_gd = 0.0
        d_norm = np.linalg.norm(self.d)
        g_norm = np.linalg.norm(self.g_new)
        if d_norm > 1e-12 and g_norm > 1e-12:
            cos_gd = np.dot(self.g_new, self.d) / (d_norm * g_norm)

        return {
            "log_eta": np.array([np.log(self.eta)], dtype=np.float32),
            "log_delta": np.array([current_log_delta], dtype=np.float32),
            "serious_step": np.array([1.0 if self.serious_step else 0.0], dtype=np.float32),
            "d_norm_sq": np.array([d_norm ** 2], dtype=np.float32),
            "proximal_term": np.array([self.eta * (d_norm ** 2)], dtype=np.float32),
            "lin_error": np.array([self.lin_error], dtype=np.float32),
            "gap_improve_1": np.array([gap_improve_1], dtype=np.float32),
            "gap_improve_2": np.array([gap_improve_2], dtype=np.float32),
            "gap_improve_3": np.array([gap_improve_3], dtype=np.float32),
            "g_norm_sq": np.array([g_norm ** 2], dtype=np.float32),
            "cos_gd": np.array([cos_gd], dtype=np.float32)
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
            verbose=verbose
        )
        return env, None
