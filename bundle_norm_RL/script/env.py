"""
Level Bundle RL Environment

使用 RL 直接预测乘子 (pi, pi0)，替代 Level Bundle 中的 outer problem QP 求解。

State:
    - subgradient_history: 最近 K 轮的次梯度 (K, N_VARS+1)
    - valid_mask: 有效次梯度掩码 (K,)
    - pi: 当前乘子 pi (N_VARS,)
    - pi0: 当前乘子 pi0 (1,)
    - lb_ub_norm: [LB_norm, UB_norm, gap_norm] (3,) 归一化后，避免 NaN
    - trial_point: 展平的 trial point (trial_point_dim,)
    - realization: 场景特征 (realization_dim,)

Action:
    - action: (N_VARS+1,) ∈ [-1, 1]
      SB3 SquashedDiagGaussian 内置 tanh + log_prob 校正
      pi 直接用 action ∈ [-1, 1]
      pi0 = (action + 1) / 2 ∈ [0, 1]，保证非负

Reward:
    - dual 值差分 / running_std，clip 到 [-3, 3]
    - running mean/std 跨 episode 持久化（Welford 在线算法）

UB 计算:
    - 训练时 (use_outer=True): 使用 OuterProblem 求解真实 UB
    - 测试时 (use_outer=False): 不计算 UB，UB 设为 LB（仅看 LB 收敛）
"""

import gymnasium as gym
import numpy as np
from bundle_norm_RL.script.level_bundle_problem import InnerProblem, OuterProblem


class LevelBundleEnv(gym.Env):
    # 跨 episode 持久化的 reward 归一化统计
    _global_reward_mean = 0.0
    _global_reward_var = 1.0
    _global_reward_count = 0

    def __init__(self, logger, config, n, K=20, verbose=False, use_outer=True):
        """
        Args:
            logger: 日志器
            config: LevelBundleConfig 对象
            n: realization 索引
            K: 次梯度历史长度
            verbose: 是否输出详细日志
            use_outer: 是否使用 OuterProblem 计算 UB（训练时 True，测试时 False）
        """
        super().__init__()

        self.logger = logger
        self.config = config
        self.n = n
        self.K = K
        self.verbose = verbose
        self.use_outer = use_outer

        self.problem_params = config.PROBLEM_PARAMS
        self.stage = config.T
        self.N_VARS = config.N_VARS

        # inner problem
        self.inner_problem = InnerProblem(logger, config, n)

        # outer problem（仅训练时使用）
        self.outer_problem = None
        if self.use_outer:
            self.outer_problem = OuterProblem(
                logger,
                dim_pi=config.N_VARS,
                X_trial=config.X_trial,
                theta_trial=float(config.THETA_TRIAL),
            )

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
            "lb_ub_norm": gym.spaces.Box(
                low=-10, high=10, shape=(3,), dtype=np.float32
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

        # 动作空间: [-1, 1]，SB3 使用 SquashedDiagGaussian（内置 tanh + log_prob 校正）
        # pi 直接用 action（SB3 的 tanh squash 已映射到 [-1,1]）
        # pi0 = (action + 1) / 2 ∈ [0, 1]，保证非负
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.N_VARS + 1,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.pi = np.zeros(self.N_VARS, dtype=np.float32)
        self.pi0 = 0.001
        self.LB = 0.0
        self.UB = 0.0
        self.t = 0

        # 重建 outer problem（每次 reset 清空旧的 cuts）
        if self.use_outer:
            self.outer_problem = OuterProblem(
                self.logger,
                dim_pi=self.config.N_VARS,
                X_trial=self.config.X_trial,
                theta_trial=float(self.config.THETA_TRIAL),
            )

        # 初始求解
        z_X_values, obj_term_value, inner_obj = self.inner_problem.solve(
            self.pi, self.pi0
        )

        if z_X_values is None:
            raise RuntimeError("Inner model failed at reset")

        # 初始 LB
        self.LB = inner_obj - self.pi @ self.X_trial - self.pi0 * self.theta_trial

        # 缩放因子：使用 LB 绝对值作为量级参考
        self.scale = abs(self.LB) + 1.0
        # 记录初始 LB 用于归一化
        self.LB_init = self.LB

        # 初始 dual 值（上一步的 dual，用于计算 reward 差分）
        self.prev_dual = self.LB

        # 初始次梯度 → 加入 outer problem
        subgradient = np.array(z_X_values + [obj_term_value], dtype=np.float32)
        self.subgradient_list = [subgradient]

        if self.use_outer:
            self.outer_problem.add_cut(subgradient.tolist())
            _, _, self.UB = self.outer_problem.solve()
            if self.UB is None:
                self.UB = self.LB * 1.1  # fallback
        else:
            self.UB = self.LB  # 测试时不计算 UB

        return self._get_state(), {}

    def step(self, action):
        """
        Action: raw (pi_raw[N_VARS], pi0_raw[1])
        后处理: pi = tanh(raw_pi), pi0 = softplus(raw_pi0)
        """
        action = np.asarray(action, dtype=np.float32).copy()

        # pi 直接用 action（SB3 SquashedDiagGaussian 已将高斯映射到 [-1,1]）
        pi = action[:self.N_VARS]
        # pi0 = (action + 1) / 2 ∈ [0, 1]，保证非负
        pi0 = (action[self.N_VARS] + 1.0) / 2.0 + 1e-6

        self.pi = pi
        self.pi0 = pi0

        # 求解 inner problem
        z_X_values, obj_term_value, inner_obj = self.inner_problem.solve(pi, pi0)

        if z_X_values is None:
            reward = -0.5
        else:
            # 计算 dual 值: inner_obj - pi^T * X_trial - pi0 * theta_trial
            dual = inner_obj - pi @ self.X_trial - pi0 * self.theta_trial

            # reward = dual 相对于上一步的差分
            raw_reward = dual - self.prev_dual

            # 跨 episode 的 running mean/std 归一化（Welford 在线算法）
            LevelBundleEnv._global_reward_count += 1
            delta = raw_reward - LevelBundleEnv._global_reward_mean
            LevelBundleEnv._global_reward_mean += delta / LevelBundleEnv._global_reward_count
            delta2 = raw_reward - LevelBundleEnv._global_reward_mean
            LevelBundleEnv._global_reward_var += delta * delta2
            reward_std = max(
                (LevelBundleEnv._global_reward_var / LevelBundleEnv._global_reward_count) ** 0.5,
                1.0
            )
            reward = raw_reward / reward_std

            # clip 防止极端值
            reward = np.clip(reward, -3.0, 3.0)

            # 更新历史最优 LB
            if dual > self.LB:
                self.LB = dual

            # 记录当前 dual 供下一步差分
            self.prev_dual = dual

            # 记录次梯度
            subgradient = np.array(z_X_values + [obj_term_value], dtype=np.float32)
            self.subgradient_list.append(subgradient)

            # 更新 UB：通过 outer problem 求解
            if self.use_outer:
                self.outer_problem.add_cut(subgradient.tolist())
                _, _, new_UB = self.outer_problem.solve()
                if new_UB is not None:
                    self.UB = new_UB
            # 测试时不更新 UB

        self.t += 1
        terminated = self.t >= self.K

        if self.verbose:
            self.logger.debug(
                f"[LevelBundleEnv Step {self.t}] "
                f"pi0={pi0:.6f}, pi_norm={np.linalg.norm(pi):.6f}, "
                f"LB={self.LB:.4f}, UB={self.UB:.4f}, reward={reward:.6f}"
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

        # 归一化 LB/UB/gap，避免 NaN 和极端值
        lb_norm = (self.LB - self.LB_init) / self.scale
        ub_norm = (self.UB - self.LB_init) / self.scale if self.use_outer else lb_norm
        gap_norm = (self.UB - self.LB) / self.scale if self.use_outer else 0.0
        lb_ub_norm = np.array([lb_norm, ub_norm, gap_norm], dtype=np.float32)

        realization_feature = np.concatenate([self.p_d, self.re])

        return {
            "subgradient_history": history,
            "valid_mask": valid_mask,
            "pi": self.pi.astype(np.float32),
            "pi0": np.array([self.pi0], dtype=np.float32),
            "lb_ub_norm": lb_ub_norm,
            "trial_point": self.X_trial,
            "realization": realization_feature,
        }

    @classmethod
    def create_env(cls, logger, config, K=20, verbose=False, use_outer=True):
        """创建环境"""
        env = cls(
            logger=logger,
            config=config,
            n=config.n,
            K=K,
            verbose=verbose,
            use_outer=use_outer,
        )
        return env
