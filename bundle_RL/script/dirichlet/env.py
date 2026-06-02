import gymnasium as gym
import numpy as np
from bundle_RL.script.lag_problem import SubProblem

"""
Attention-based Bundle Environment

state：当前的所有cuts，valid_mask，当前的pi值，当前的trial_point、以及场景 realization
action：lambda 和 步长

状态转移：lambda + 步长 -> 归一化 -> pi -> sub求解得到子问题

reward：Log-Gap: log(gap_old) - log(gap_new)，其中 gap = max(0, -phi)

特点:
1. 使用 valid_mask 标记有效的 cuts
2. bundle 数据结构预留 age 和 error 字段
"""


class BundleDualEnv(gym.Env):
    def __init__(self, logger, config, n, state_dim, K, verbose=False):
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
        
        # 计算 realization 特征维度（不包含 prob）
        self.realization_dim = len(self.p_d) + len(self.re)  # p_d + re

        # ========== 状态空间 ==========
        # shape = (K, state_dim)
        # 使用Box，padding部分为0
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
        # 前K维是lambda，最后1维是步长
        # 使用标准范围 [-1, 1]，在 step 方法中进行缩放
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

        self.bundle = []
        self.pi = np.zeros(self.state_dim)  # 初始化pi
        self.t = 0  # 迭代次数

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

        return self._get_state(), {}

    # --------------------------------------------------

    def step(self, action):
        """
        action = [lambda_1 ... lambda_K , eta]
        """
        # 拆分动作
        raw_lambda = action[:self.K]
        raw_eta = action[-1]

        # ---------- lambda 归一化 ----------
        exp_lambda = np.exp(raw_lambda - np.max(raw_lambda))
        lambdas = exp_lambda / (np.sum(exp_lambda) + 1e-8)

        # ---------- 步长映射 ----------
        # 用sigmoid保证正值，并限制最大步长
        eta = 1.0 * (1 / (1 + np.exp(-raw_eta)))

        # ---------- 用 state 聚合 ----------
        state = self._get_state()
        G = state["cuts"]
        valid_mask = state["valid_mask"]

        # 使用 mask 过滤无效 cuts
        masked_lambdas = lambdas * valid_mask
        lambda_sum = np.sum(masked_lambdas) + 1e-8
        normalized_lambdas = masked_lambdas / lambda_sum

        d = normalized_lambdas @ G  # (state_dim,)

        # 更新pi
        self.pi = self.pi + eta * d

        # 子问题求解
        g, phi_new = self.subproblem.solve(self.pi)

        # ========== Log-Gap Reward ==========
        # gap = max(0, -phi)，对于最小化问题，gap 越小越好
        # log-gap reward: gap 减小时为正，gap 增大时为负
        # phi_old = self.bundle[-1]["phi"]
        # eps = 1e-8
        # gap_old = max(0, -phi_old) + eps
        # gap_new = max(0, -phi_new) + eps
        #
        # raw_reward = np.log(gap_old) - np.log(gap_new)
        #
        # # 归一化: 使用 tanh 压缩到合理范围
        # reward = np.tanh(raw_reward * 0.1)  # 缩放系数可调

        # reward 使用子问题的目标函数的提升值
        reward = (phi_new - self.bundle[-1]["phi"]) / self.scale

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
        
        # 记录每次step的输出值（仅在verbose模式下）
        if self.verbose:
            self.logger.debug(f"[BundleEnv Step {self.t}] "
                             f"raw_eta={raw_eta:.4f}, "
                             f"eta={eta:.4f}, "
                             f"pi_norm={np.linalg.norm(self.pi):.6f}, "
                             f"phi_new={phi_new:.6f}, "
                             # f"gap_new={gap_new:.6f}, "
                             # f"raw_reward={raw_reward:.6f}, "
                             f"reward={reward:.6f}, "
                             f"active_cuts={int(np.sum(valid_mask))}, "
                             f"terminated={terminated}")

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
        from bundle_RL.script.lag_problem import MasterProblem

        env = cls(
            logger=logger,
            config=config,
            n=config.n,
            state_dim=config.N_VARS,
            K=K,
            verbose=verbose
        )
        master = MasterProblem(logger, config.N_VARS, tolerance=tolerance)
        return env, master
