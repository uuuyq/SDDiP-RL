import gymnasium as gym
import numpy as np
from lag_problem import SubProblem
from features import bundle_features


"""
state：当前的所有cuts，当前的pi值
action：lambda 和 步长

状态转移：lambda + 步长 -> 归一化 -> pi -> sub求解得到子问题

reward：pi对应的子问题最优解对应的目标函数值，求解的真实值，让子问题的解尽可能大

"""

class ProblemData:
    def __init__(self, logger, problem_params, trial_point, t, n, i):
        self.logger = logger
        self.problem_params = problem_params
        self.trial_point = trial_point
        self.t = t
        self.n = n
        self.i = i


class BundleDualEnv(gym.Env):
    def __init__(self, problemData, state_dim, K):
        """

        :param problemData: 构建子问题的参数
        :param state_dim: cut中次梯度的维度
        :param K: 使用padding的方式，K代表最大的cuts数，同时也是输出的lambda维度
        """
        super().__init__()

        self.subproblem = SubProblem(
            problemData.logger,
            problemData.problem_params,
            problemData.trial_point,
            problemData.t,
            problemData.n,
            problemData.i
        )
        self.K = K
        self.state_dim = state_dim
        self.action_dim = K + 1  # 输出lambda以及步长

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.bundle = []
        self.pi = np.zeros(self.state_dim)
        self.t = 0  # 迭代次数

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
        action = [lambda_1 ... lambda_K , eta]
        """
        # 拆分动作
        raw_lambda = action[:self.K]
        raw_eta = action[-1]

        # ---------- lambda 归一化 ----------
        exp_lambda = np.exp(raw_lambda)
        lambdas = exp_lambda / (np.sum(exp_lambda) + 1e-8)

        # ---------- 步长映射 ----------
        # 用sigmoid保证正值，并限制最大步长
        # TODO: 步长的上界具体设置可以查看bundle算法中的步长大小
        eta = 1.0 * (1 / (1 + np.exp(-raw_eta)))

        # ---------- 用 state 聚合 ----------
        state = self._get_state()
        G = state["cuts"]
        d = lambdas @ G  # (state_dim,)

        # 更新pi
        self.pi = self.pi + eta * d

        # 子问题求解
        g, phi_new = self.subproblem.solve(self.pi)

        cut_new = {
            "pi": self.pi.copy(),
            "g": g.copy(),
            "phi": phi_new,
        }

        # reward 使用子问题的目标函数的提升值
        reward = (phi_new - self.bundle[-1]["phi"]) / self.scale
        self.bundle.append(cut_new)

        self.t += 1
        terminated = self.t >= self.K

        return self._get_state(), reward, terminated, False, {}

    # --------------------------------------------------
    def _get_state(self):
        """
        获取当前最新的状态，从self.bundle中抽取最新的数据，padding出cuts矩阵
        :return: cuts，pi
        """
        cuts = np.zeros((self.K, self.state_dim), dtype=np.float32)
        # 取出最后K个最新数据（为了应对迭代次数超过K的情况，丢弃旧数据）
        active = self.bundle[-self.K:]

        start = self.K - len(active)

        for i, cut in enumerate(active):
            cuts[start + i] = cut["g"]

        return {
            "cuts": cuts,
            "pi": self.pi.astype(np.float32)
        }

    #
    # def reset(self):
    #     self.t = 0
    #     self.pi = np.zeros(10)
    #     self.bundle = []
    #     self.w_prev = np.zeros_like(self.pi)
    #     self.eta_prev = 1.0
    #
    #     phi, g = self.subproblem.solve(self.pi)
    #     self.bundle.append((self.pi.copy(), g, phi))
    #
    #     return self._state()
    #
    # def step(self, lambdas):
    #     lambdas = lambdas / (np.sum(lambdas) + 1e-8)
    #
    #     active = self.bundle[-self.K:]
    #     w = np.zeros_like(self.pi)
    #
    #     for l, (_, g, _) in zip(lambdas, active):
    #         w += l * g
    #
    #     self.pi = self.pi + self.eta_prev * w
    #
    #     phi_new, g_new = self.subproblem.solve(self.pi)
    #     self.bundle.append((self.pi.copy(), g_new, phi_new))
    #
    #     reward = phi_new   # 外层 max
    #
    #     self.w_prev = w
    #     self.t += 1
    #
    #     done = self.t >= self.T
    #
    #     return self._state(), reward, done, {"phi": phi_new}
    #
    # def _state(self):
    #     return bundle_features(
    #         self.bundle,
    #         self.pi,
    #         self.w_prev,
    #         self.eta_prev,
    #         self.t
    #     )
