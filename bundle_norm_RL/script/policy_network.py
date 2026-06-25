"""
Custom Actor-Critic Policy for Level Bundle RL

直接输出 (pi_raw, pi0_raw)，环境侧做归一化。

网络结构:
    FeaturesExtractor → [seq_emb; global_emb] (B, 2H)
        ├── ActorHead → action_mean (B, N_VARS+1)
        └── ValueHead → V(s) (B, 1) — 更深的网络 + LayerNorm
"""

import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces
from typing import Dict, Tuple

from bundle_norm_RL.script.features_extractor import LevelBundleFeaturesExtractor


class ActorHead(nn.Module):
    """
    Actor Head: 输出 action_mean (B, N_VARS+1)
    """
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            # 不加 tanh：环境侧做归一化，action_space [-1,1] 由 SB3 的 squash 处理
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class ValueHead(nn.Module):
    """
    Value Head: 输出 V(s)，比 Actor 更深以增加容量

    Critic 需要更强的拟合能力，因为 V(s) 需要预测累积回报
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class LevelBundleActorCriticPolicy(ActorCriticPolicy):
    """
    自定义 Actor-Critic 策略

    继承 SB3 的 ActorCriticPolicy，复用 action_dist 和 log_std，
    重写 forward/evaluate_actions/predict_values/_predict。
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        lr_schedule,
        features_extractor_class=LevelBundleFeaturesExtractor,
        features_extractor_kwargs=None,
        **kwargs,
    ):
        if features_extractor_kwargs is None:
            features_extractor_kwargs = dict(hidden_dim=64)

        if "net_arch" not in kwargs:
            kwargs["net_arch"] = []

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            **kwargs,
        )

        hidden_dim = self.features_extractor.hidden_dim
        combined_dim = 2 * hidden_dim
        action_dim = action_space.shape[0]

        self.actor_head = ActorHead(combined_dim, action_dim, hidden_dim)
        self.value_head = ValueHead(combined_dim, hidden_dim)

        # log_std 初始化为较小值，控制初始探索幅度
        # log_std=-2 → std≈0.135
        with torch.no_grad():
            self.log_std.fill_(-2.0)

        # 重建优化器以包含新 head
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _encode(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """获取 combined embedding (B, 2H)"""
        return self.features_extractor(obs)

    def _compute_action_mean(self, h: torch.Tensor) -> torch.Tensor:
        return self.actor_head(h)

    def forward(
        self, obs: Dict[str, torch.Tensor], deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self._encode(obs)
        action_mean = self._compute_action_mean(h)
        distribution = self.action_dist.proba_distribution(action_mean, self.log_std)

        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_head(h)

        return actions, values, log_prob

    def evaluate_actions(
        self, obs: Dict[str, torch.Tensor], actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self._encode(obs)
        action_mean = self._compute_action_mean(h)
        distribution = self.action_dist.proba_distribution(action_mean, self.log_std)

        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.value_head(h)

        return values, log_prob, entropy

    def predict_values(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        h = self._encode(obs)
        return self.value_head(h)

    def _predict(
        self, observation: Dict[str, torch.Tensor], deterministic: bool = False
    ) -> torch.Tensor:
        h = self._encode(observation)
        action_mean = self._compute_action_mean(h)
        distribution = self.action_dist.proba_distribution(action_mean, self.log_std)
        return distribution.get_actions(deterministic=deterministic)
