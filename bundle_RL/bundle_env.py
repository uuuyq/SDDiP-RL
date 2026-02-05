import gymnasium as gym
import numpy as np
from lag_problem import Subproblem
from features import bundle_features

class BundleDualEnv(gym.Env):
    def __init__(self, problem_data, K=10, T=20):
        super().__init__()
        self.K = K
        self.T = T

        self.subproblem = Subproblem(problem_data)

        self.state_dim = 18
        self.action_dim = K

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,)
        )

        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(K,)
        )

        self.reset()

    def reset(self):
        self.t = 0
        self.pi = np.zeros(10)
        self.bundle = []
        self.w_prev = np.zeros_like(self.pi)
        self.eta_prev = 1.0

        phi, g = self.subproblem.solve(self.pi)
        self.bundle.append((self.pi.copy(), g, phi))

        return self._state()

    def step(self, lambdas):
        lambdas = lambdas / (np.sum(lambdas) + 1e-8)

        active = self.bundle[-self.K:]
        w = np.zeros_like(self.pi)

        for l, (_, g, _) in zip(lambdas, active):
            w += l * g

        self.pi = self.pi + self.eta_prev * w

        phi_new, g_new = self.subproblem.solve(self.pi)
        self.bundle.append((self.pi.copy(), g_new, phi_new))

        reward = phi_new   # 外层 max

        self.w_prev = w
        self.t += 1

        done = self.t >= self.T

        return self._state(), reward, done, {"phi": phi_new}

    def _state(self):
        return bundle_features(
            self.bundle,
            self.pi,
            self.w_prev,
            self.eta_prev,
            self.t
        )
