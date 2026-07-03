"""
Incremental Level Bundle RL Environment

使用 RL 预测增量 d = (d_pi, d_pi0)，替代 Level Bundle 增量形式中的 Level QP 求解。

与 script/env.py 的区别:
    - Action: 预测增量 d = (d_pi, d_pi0)，而非绝对乘子 (pi, pi0)
    - 需要维护稳定中心 (pi_bar, pi0_bar) 和 Serious Step 逻辑
    - 使用 IncrementalOuterProblem 替代 OuterProblem
    - 状态包含稳定中心信息

State:
    - subgradient_history: 最近 K 轮的次梯度 + 生成该次梯度时的乘子 (K, 2*(N_VARS+1))
      每行 = [subgradient_pi(N_VARS), subgradient_pi0(1), gen_pi(N_VARS), gen_pi0(1)]
    - valid_mask: 有效次梯度掩码 (K,)
    - pi_bar: 当前稳定中心 pi (N_VARS,)
    - pi0_bar: 当前稳定中心 pi0 (1,)
    - pi: 当前乘子 pi (N_VARS,)
    - pi0: 当前乘子 pi0 (1,)
    - lb_ub_norm: [LB_norm, UB_norm, gap_norm] (3,)
    - trial_point: 展平的 trial point (trial_point_dim,)
    - realization: 场景特征 (realization_dim,)

Action:
    - action: (N_VARS+1,) ∈ [-1, 1]
      d_pi = action[:N_VARS] * scale_d  (缩放到合适范围)
      d_pi0 = action[N_VARS] * scale_d_pi0

Reward:
    - dual / scale，scale = |LB_init| + 1
    - 迭代惩罚 (-0.05) 和收敛奖励

UB 计算:
    - 训练时 (use_outer=True): 使用 IncrementalOuterProblem 求解真实 UB
    - 测试时 (use_outer=False): 不计算 UB，UB 设为 LB
"""

import gymnasium as gym
import numpy as np
from bundle_norm_RL.script1.level_bundle_problem import InnerProblem, IncrementalOuterProblem


class IncrementalLevelBundleEnv(gym.Env):

    def __init__(self, logger, config, n, K=20, verbose=False, use_outer=True):
        """
        Args:
            logger: 日志器
            config: LevelBundleConfig 对象
            n: realization 索引
            K: 次梯度历史长度
            verbose: 是否输出详细日志
            use_outer: 是否使用 IncrementalOuterProblem 计算 UB（训练时 True，测试时 False）
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
            self.outer_problem = IncrementalOuterProblem(
                logger,
                dim_pi=config.N_VARS,
                X_trial=config.X_trial,
                theta_trial=float(config.THETA_TRIAL),
                rho=config.rho,
                B_t=config.B_t,
                norm_bound_type=config.norm_bound_type,
                weights=config.weights,
            )

        # trial point
        self.X_trial = np.array(config.X_trial, dtype=np.float32)
        self.theta_trial = float(config.THETA_TRIAL)
        self.trial_point_dim = len(self.X_trial)

        # realization 特征
        self.p_d = np.array(self.problem_params.p_d[self.stage][self.n], dtype=np.float32)
        self.re = np.array(self.problem_params.re[self.stage][self.n], dtype=np.float32)
        self.realization_dim = len(self.p_d) + len(self.re)

        # 增量缩放因子 (根据 analyze_increments 的统计确定)
        self.scale_d = 0.1       # d_pi 缩放因子
        self.scale_d_pi0 = 0.05  # d_pi0 缩放因子

        # 状态空间
        # subgradient_history 每行 = [subgradient(N_VARS+1), gen_pi(N_VARS), gen_pi0(1)]
        self.CUT_DIM = 2 * (self.N_VARS + 1)

        self.observation_space = gym.spaces.Dict({
            "subgradient_history": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.K, self.CUT_DIM), dtype=np.float32
            ),
            "valid_mask": gym.spaces.Box(
                low=0, high=1, shape=(self.K,), dtype=np.float32
            ),
            "pi_bar": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.N_VARS,), dtype=np.float32
            ),
            "pi0_bar": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
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
        # d_pi = action[:N_VARS] * scale_d
        # d_pi0 = action[N_VARS] * scale_d_pi0
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.N_VARS + 1,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 初始化乘子和稳定中心
        self.pi = np.zeros(self.N_VARS, dtype=np.float32)
        self.pi0 = 0.1
        self.pi_bar = np.zeros(self.N_VARS, dtype=np.float32)
        self.pi0_bar = 0.1

        self.LB = 0.0
        self.UB = 0.0
        self.current_dual = 0.0
        self.t = 0

        # 记录每步生成 cut 时的 (pi, pi0)
        self.pi_history = []
        self.pi0_history = []

        # 重建 outer problem（每次 reset 清空旧的 cuts）
        if self.use_outer:
            self.outer_problem = IncrementalOuterProblem(
                self.logger,
                dim_pi=self.config.N_VARS,
                X_trial=self.config.X_trial,
                theta_trial=float(self.config.THETA_TRIAL),
                rho=self.config.rho,
                B_t=self.config.B_t,
                norm_bound_type=self.config.norm_bound_type,
                weights=self.config.weights,
            )

        # 初始求解
        z_X_values, obj_term_value, inner_obj = self.inner_problem.solve(
            self.pi, self.pi0
        )

        if z_X_values is None:
            raise RuntimeError("Inner model failed at reset")

        # 初始 LB
        self.LB = inner_obj - self.pi @ self.X_trial - self.pi0 * self.theta_trial
        self.current_dual = self.LB

        # 缩放因子：使用 LB 绝对值作为量级参考
        self.scale = abs(self.LB) + 1.0
        # 记录初始 LB 用于归一化
        self.LB_init = self.LB

        # 初始次梯度 → 加入 outer problem
        subgradient = np.array(z_X_values + [obj_term_value], dtype=np.float32)
        self.subgradient_list = [subgradient]
        # 记录生成该 cut 时的 (pi, pi0)
        self.pi_history.append(self.pi.copy())
        self.pi0_history.append(self.pi0)

        if self.use_outer:
            lambda_i = np.concatenate([self.pi, [self.pi0]])
            omega_i = inner_obj - self.pi @ self.X_trial - self.pi0 * self.theta_trial
            self.outer_problem.add_cut(subgradient.tolist(), lambda_i, omega_i)
            _, _, self.UB = self.outer_problem.solve()
            if self.UB is None:
                self.UB = self.LB * 1.1  # fallback
        else:
            self.UB = self.LB  # 测试时不计算 UB

        return self._get_state(), {}

    def step(self, action):
        """
        Action: raw action (N_VARS+1,) ∈ [-1, 1]
        后处理: d_pi = action[:N_VARS] * scale_d
                 d_pi0 = action[N_VARS] * scale_d_pi0
        新乘子: pi = pi_bar + d_pi, pi0 = pi0_bar + d_pi0
        """
        action = np.asarray(action, dtype=np.float32).copy()

        # 解码增量
        d_pi = action[:self.N_VARS] * self.scale_d
        d_pi0 = action[self.N_VARS] * self.scale_d_pi0

        # 恢复乘子: λ = λ̄ + d
        pi = self.pi_bar + d_pi
        pi0 = self.pi0_bar + d_pi0
        # 保证 pi0 非负
        pi0 = max(pi0, 1e-6)

        self.pi = pi
        self.pi0 = pi0

        # 求解 inner problem
        z_X_values, obj_term_value, inner_obj = self.inner_problem.solve(pi, pi0)

        if z_X_values is None:
            reward = -1.0 / self.scale
        else:
            # 计算 dual 值: inner_obj - pi^T * X_trial - pi0 * theta_trial
            dual = inner_obj - pi @ self.X_trial - pi0 * self.theta_trial

            # reward = dual / scale
            reward = dual / self.scale

            # 更新当前迭代的实际 dual 值
            self.current_dual = dual

            # 更新历史最优 LB
            if dual > self.LB:
                self.LB = dual

            # 记录次梯度
            subgradient = np.array(z_X_values + [obj_term_value], dtype=np.float32)
            self.subgradient_list.append(subgradient)
            # 记录生成该 cut 时的 (pi, pi0)
            self.pi_history.append(self.pi.copy())
            self.pi0_history.append(self.pi0)

            # 更新 UB：通过 outer problem 求解
            if self.use_outer:
                lambda_i = np.concatenate([pi, [pi0]])
                omega_i = dual  # inner_obj - pi^T * X_trial - pi0 * theta_trial
                self.outer_problem.add_cut(subgradient.tolist(), lambda_i, omega_i)
                _, _, new_UB = self.outer_problem.solve()
                if new_UB is not None:
                    self.UB = new_UB

            # Serious step: 当 dual 有显著提升时，更新稳定中心
            if dual > self.LB - 1e-8 * max(1.0, abs(self.LB)):
                # 只要当前 dual 不比 LB 差太多，就考虑更新稳定中心
                if dual > self.LB + 1e-8 * max(1.0, abs(self.LB)):
                    self.pi_bar = pi.copy()
                    self.pi0_bar = pi0
                    if self.use_outer:
                        self.outer_problem.set_stability_center(pi.copy(), pi0)

        self.t += 1

        # 迭代惩罚
        reward -= 0.05

        # 终止条件
        max_iter_reached = self.t >= self.K
        gap_converged = False
        if self.use_outer and self.UB is not None:
            gap = self.UB - self.LB
            if gap < self.config.gap_tol * abs(self.UB) or gap < 1e-6:
                gap_converged = True
                remaining_steps = self.K - self.t
                reward += remaining_steps * 0.1
        terminated = max_iter_reached or gap_converged

        if self.verbose:
            pi_str = " ".join(f"{p:.6f}" for p in pi)
            d_pi_str = " ".join(f"{d:.6f}" for d in d_pi)
            self.logger.debug(
                f"[IncrementalEnv Step {self.t}] "
                f"pi0={pi0:.6f}, pi={pi_str}, "
                f"d_pi0={d_pi0:.6f}, d_pi={d_pi_str}, "
                f"LB={self.LB:.4f}, UB={self.UB:.4f}, reward={reward:.6f}"
            )

        return self._get_state(), float(reward), terminated, False, {}

    def _get_state(self):
        """构建当前状态"""
        # 次梯度历史 + 生成乘子历史，取最近 K 个
        history = np.zeros((self.K, self.CUT_DIM), dtype=np.float32)
        valid_mask = np.zeros(self.K, dtype=np.float32)

        active = self.subgradient_list[-self.K:]
        active_pi = self.pi_history[-self.K:]
        active_pi0 = self.pi0_history[-self.K:]
        start = self.K - len(active)
        for i in range(len(active)):
            history[start + i, :self.N_VARS + 1] = active[i]
            history[start + i, self.N_VARS + 1:2 * self.N_VARS + 1] = active_pi[i]
            history[start + i, -1] = active_pi0[i]
            valid_mask[start + i] = 1.0

        # 归一化 LB/UB/gap
        lb_norm = (self.LB - self.LB_init) / self.scale
        ub_norm = (self.UB - self.LB_init) / self.scale if self.use_outer else lb_norm
        gap_norm = (self.UB - self.LB) / self.scale if self.use_outer else 0.0
        lb_ub_norm = np.array([lb_norm, ub_norm, gap_norm], dtype=np.float32)

        realization_feature = np.concatenate([self.p_d, self.re])

        return {
            "subgradient_history": history,
            "valid_mask": valid_mask,
            "pi_bar": self.pi_bar.astype(np.float32),
            "pi0_bar": np.array([self.pi0_bar], dtype=np.float32),
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
