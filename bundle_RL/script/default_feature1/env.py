import gymnasium as gym
import numpy as np
from bundle_RL.script.lag_problem import SubProblem

"""
在 default_feature 基础上进一步调整：
- 不输出 eta 步长，使用固定步长 0.5
- 增加 state：
  - valid_mask (仅内部使用，不作为输入)
  - search_direction_norm (上一次d的二范数)
  - linear_improvement (截距项加权和)
  - cosine_sim (最新g_new与d的余弦相似度)
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
        self.action_dim = K  # 只输出lambda，不再输出eta
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
            ),
            "search_direction_norm": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1,),
                dtype=np.float32
            ),
            # "linear_improvement": gym.spaces.Box(
            #     low=-np.inf,
            #     high=np.inf,
            #     shape=(1,),
            #     dtype=np.float32
            # ),
            "cosine_sim": gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            ),
            "bundle_ratio": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            ),

        })

        # ========== 动作空间 ==========
        # 输出lambda
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
        self.last_d = np.zeros(self.state_dim)  # 上一次的d
        self.last_lambdas = np.zeros(self.K)  # 上一次的lambda

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

        return self._get_state(), {}

    # --------------------------------------------------

    def step(self, action):
        """
        action = [lambda_1 ... lambda_K]
        """
        # 获取 raw_lambda
        raw_lambda = action[:self.K]

        # ---------- 计算 valid_mask ----------
        # valid_mask: True 表示该位置是有效的 cut，False 表示是 padding
        # 有效 cut 在最前面，padding 在后面
        valid_mask = np.zeros(self.K, dtype=bool)
        num_active = min(len(self.bundle), self.K)
        valid_mask[:num_active] = True

        # ---------- lambda 归一化（使用 valid_mask 屏蔽 padding） ----------
        # 先对 padding 位置的 raw_lambda 减去一个很大的值，使得 exp 后接近 0
        masked_raw_lambda = raw_lambda.copy()
        masked_raw_lambda[~valid_mask] = -1e10
        
        exp_lambda = np.exp(masked_raw_lambda)
        lambdas = exp_lambda / (np.sum(exp_lambda) + 1e-8)

        # ---------- 固定步长 ----------
        eta = 0.5

        # ---------- 用 state 聚合 ----------
        state = self._get_state()
        G = state["cuts"]
        d = lambdas @ G  # (state_dim,)

        print("############bundle_RL#########")
        print("action = ", action)
        print("lambda = ", lambdas)
        print("eta = ", eta)

        # 更新 pi
        self.pi = self.pi + eta * d

        # 子问题求解
        g, phi_new = self.subproblem.solve(self.pi)

        cut_new = {
            "pi": self.pi.copy(),
            "g": g.copy(),
            "phi": phi_new,
        }

        # reward 使用子问题的目标函数的提升值
        # reward = (phi_new - self.bundle[-1]["phi"]) / self.scale
        reward = np.log(phi_new) - np.log(self.bundle[-1]["phi"])
        print(f"reward: {reward}")
        self.bundle.append(cut_new)

        # 保存当前的d和lambda用于下一次计算新特征
        self.last_d = d.copy()
        self.last_lambdas = lambdas.copy()

        self.t += 1
        terminated = self.t >= self.K
        
        # 记录每次step的输出值（仅在verbose模式下）
        if self.verbose:
            self.logger.debug(f"[BundleEnv Step {self.t}] "
                             f"raw_lambda={raw_lambda},"
                             f"eta={eta:.4f}, "
                             f"pi_norm={np.linalg.norm(self.pi):.6f}, "
                             f"phi_new={phi_new:.6f}, "
                             f"reward={reward:.6f}, "
                             f"terminated={terminated}")

        return self._get_state(), reward, terminated, False, {}

    # --------------------------------------------------
    def _get_state(self):
        """
        获取当前最新的状态，从self.bundle中抽取最新的数据，padding出cuts矩阵
        :return: cuts, pi, trial_point, realization, search_direction_norm, linear_improvement, cosine_sim
        """
        cuts = np.zeros((self.K, self.state_dim), dtype=np.float32)
        # 取出最后K个最新数据（为了应对迭代次数超过K的情况，丢弃旧数据）
        active = self.bundle[-self.K:]

        # 有效 cut 在最前面，padding 在后面
        for i, cut in enumerate(active):
            cuts[i] = cut["g"]

        # 构建 realization 特征向量（不包含 prob）
        realization_feature = np.concatenate([
            self.p_d,
            self.re,
            np.array([self.prob], dtype=np.float32)
        ])

        bundle_ratio = np.array(
            [len(active) / self.K],
            dtype=np.float32
        )

        current_phi = self.bundle[-1]["phi"]

        gap = max(-current_phi, 1e-8)


        # ========== 计算新特征 ==========
        
        # 1. search_direction_norm：上一次d的二范数
        search_direction_norm = np.array([np.linalg.norm(self.last_d)], dtype=np.float32)
        
        # 2. linear_improvement：截距项加权和
        # 截距项 = f_new - g_new · pi_new
        # linear_improvement = 0.0
        # if len(self.bundle) > 0 and len(active) > 0:
        #     # 使用上一次的lambda对当前active cuts的截距项加权
        #     # 注意：这里需要确保lambda和cut的对应关系
        #     for i, cut in enumerate(active):
        #         intercept = cut["phi"] - np.dot(cut["g"], cut["pi"])
        #         if start + i < len(self.last_lambdas):
        #             linear_improvement += self.last_lambdas[start + i] * intercept
        # linear_improvement = np.array([linear_improvement], dtype=np.float32)
        
        # 3. cosine_sim：最新g_new与d的余弦相似度
        cosine_sim = 0.0
        if len(self.bundle) > 0:
            latest_g = self.bundle[-1]["g"]
            d_norm = np.linalg.norm(self.last_d)
            g_norm = np.linalg.norm(latest_g)
            if d_norm > 1e-12 and g_norm > 1e-12:
                cosine_sim = np.dot(latest_g, self.last_d) / (d_norm * g_norm)
            else:
                cosine_sim = 0.0
        cosine_sim = np.array([cosine_sim], dtype=np.float32)

        return {
            "cuts": cuts,
            "pi": self.pi.astype(np.float32),
            "trial_point": self.trial_point,
            "realization": realization_feature,
            "search_direction_norm": search_direction_norm,
            # "linear_improvement": linear_improvement,
            "cosine_sim": cosine_sim,
            "bundle_ratio": bundle_ratio,
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
