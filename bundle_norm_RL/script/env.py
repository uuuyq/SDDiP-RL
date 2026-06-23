"""
Level Bundle RL Environment

使用 RL 直接预测乘子 (pi, pi0)，替代 Level Bundle 中的 outer problem QP 求解。

State:
    - subgradient_history: 最近 K 轮的次梯度 (K, N_VARS+1)
    - valid_mask: 有效次梯度掩码 (K,)
    - pi: 当前乘子 pi (N_VARS,)
    - pi0: 当前乘子 pi0 (1,)
    - lb_ub: [LB, UB, gap] (3,)
    - trial_point: 展平的 trial point (trial_point_dim,)
    - realization: 场景特征 (realization_dim,)

Action:
    - raw_action: (N_VARS+1,) → 后处理归一化得到 (pi, pi0)
      pi0 = softplus(raw_pi0) 保证 > 0
      pi = raw_pi / (||raw_pi||_1 + pi0) 归一化

Reward:
    - LB 的提升量: (LB_new - LB_old) / scale
"""

import gymnasium as gym
import numpy as np
from bundle_norm_RL.script.level_bundle_problem import InnerProblem


class LevelBundleEnv(gym.Env):
    def __init__(self, logger, config, n, K=20, verbose=False):
        """
        Args:
            logger: 日志器
            config: LevelBundleConfig 对象
            n: realization 索引
            K: 次梯度历史长度
            verbose: 是否输出详细日志
        """
        super().__init__()

        self.logger = logger
        self.config = config
        self.n = n
        self.K = K
        self.verbose = verbose

        self.problem_params = config.PROBLEM_PARAMS
        self.stage = config.T
        self.N_VARS = config.N_VARS

        # inner problem
        self.inner_problem = InnerProblem(logger, config, n)

        # trial point
        self.X_trial = np.array(config.X_trial, dtype=np.float32)
        self.theta_trial = float(config.THETA_TRIAL)
        self.trial_point_dim = len(self.X_trial)

        # realization 特征
        self.p_d = np.array(self.problem_params.p_d[self.stage][self.n], dtype=np.float32)
        self.re = np.array(self.problem_params.re[self.stage][self.n], dtype=np.float32)
        self.realization_dim = len(self.p_d) + len(self.re)

        # 状态空间
        self.observation_space = gym.spaces.Dict({
            "subgradient_history": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.K, self.N_VARS + 1), dtype=np.float32
            ),
            "valid_mask": gym.spaces.Box(
                low=0, high=1, shape=(self.K,), dtype=np.float32
            ),
            "pi": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.N_VARS,), dtype=np.float32
            ),
            "pi0": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            "lb_ub": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            ),
            "trial_point": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.trial_point_dim,), dtype=np.float32
            ),
            "realization": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.realization_dim,), dtype=np.float32
            ),
        })

        # 动作空间: raw (pi_raw, pi0_raw)
        self.action_space = gym.spaces.Box(
            low=-5.0, high=5.0,
            shape=(self.N_VARS + 1,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.pi = np.zeros(self.N_VARS, dtype=np.float32)
        self.pi0 = 0.001
        self.LB = float('-inf')
        self.UB = float('inf')
        self.t = 0

        # 初始求解
        z_X_values, obj_term_value, inner_obj = self.inner_problem.solve(
            self.pi, self.pi0
        )

        if z_X_values is None:
            raise RuntimeError("Inner model failed at reset")

        # 初始 LB
        self.LB = inner_obj - self.pi @ self.X_trial - self.pi0 * self.theta_trial

        # 缩放因子
        self.scale = abs(self.LB) + 1.0

        # 次梯度历史
        subgradient = np.array(z_X_values + [obj_term_value], dtype=np.float32)
        self.subgradient_list = [subgradient]

        return self._get_state(), {}

    def step(self, action):
        """
        Action: raw (pi_raw[N_VARS], pi0_raw[1])
        后处理归一化得到合法的 (pi, pi0)
        """
        action = np.asarray(action, dtype=np.float32).copy()

        # 后处理归一化
        pi_raw = action[:self.N_VARS]
        pi0_raw = action[self.N_VARS]

        # pi0 = softplus 保证 > 0
        pi0 = np.log1p(np.exp(pi0_raw)) + 1e-6

        # 归一化: ||pi||_1 + pi0 <= 1
        l1_norm = np.sum(np.abs(pi_raw)) + pi0
        if l1_norm > 1e-8:
            pi = pi_raw / l1_norm
            pi0 = pi0 / l1_norm
        else:
            pi = np.zeros(self.N_VARS, dtype=np.float32)
            pi0 = 1.0

        self.pi = pi
        self.pi0 = pi0

        # 求解 inner problem
        z_X_values, obj_term_value, inner_obj = self.inner_problem.solve(pi, pi0)

        if z_X_values is None:
            # inner model 失败，给负奖励
            reward = -1.0
            new_LB = self.LB
        else:
            # 计算 LB
            new_LB = inner_obj - pi @ self.X_trial - pi0 * self.theta_trial

            # reward = LB 提升量
            reward = (new_LB - self.LB) / self.scale

            # 更新 LB
            if new_LB > self.LB:
                self.LB = new_LB

            # 记录次梯度
            subgradient = np.array(z_X_values + [obj_term_value], dtype=np.float32)
            self.subgradient_list.append(subgradient)

        self.t += 1
        terminated = self.t >= self.K

        # 更新 UB 估计（用当前 LB 近似，实际中需要 outer problem）
        self.UB = self.LB * 1.1  # 简单估计

        if self.verbose:
            self.logger.debug(
                f"[LevelBundleEnv Step {self.t}] "
                f"pi0={pi0:.6f}, pi_norm={np.linalg.norm(pi):.6f}, "
                f"LB={self.LB:.4f}, reward={reward:.6f}"
            )

        return self._get_state(), float(reward), terminated, False, {}

    def _get_state(self):
        """构建当前状态"""
        # 次梯度历史，取最近 K 个
        history = np.zeros((self.K, self.N_VARS + 1), dtype=np.float32)
        valid_mask = np.zeros(self.K, dtype=np.float32)

        active = self.subgradient_list[-self.K:]
        start = self.K - len(active)
        for i, sg in enumerate(active):
            history[start + i] = sg
            valid_mask[start + i] = 1.0

        gap = self.UB - self.LB if self.LB > float('-inf') else 0.0

        realization_feature = np.concatenate([self.p_d, self.re])

        return {
            "subgradient_history": history,
            "valid_mask": valid_mask,
            "pi": self.pi.astype(np.float32),
            "pi0": np.array([self.pi0], dtype=np.float32),
            "lb_ub": np.array([self.LB, self.UB, gap], dtype=np.float32),
            "trial_point": self.X_trial,
            "realization": realization_feature,
        }

    @classmethod
    def create_env(cls, logger, config, K=20, verbose=False):
        """创建环境"""
        env = cls(
            logger=logger,
            config=config,
            n=config.n,
            K=K,
            verbose=verbose,
        )
        return env
