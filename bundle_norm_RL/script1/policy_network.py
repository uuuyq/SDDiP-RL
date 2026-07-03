"""
Custom Actor-Critic Policy for Incremental Level Bundle RL

与 script/policy_network.py 的区别:
    - 使用 IncrementalLevelBundleFeaturesExtractor
    - 动作含义为增量 d = (d_pi, d_pi0)，而非绝对乘子

网络结构:
    FeaturesExtractor → [seq_emb; global_emb] (B, 2H)
        ├── ActorHead → action_mean (B, N_VARS+1)
        └── ValueHead → V(s) (B, 1)

    cut_aware 模式:
        FeaturesExtractor → [seq_emb; global_emb] (B, 2H)
        + candidate_actions (B, K, N_VARS+1) + attention_weights (B, K)
        ├── CutAwareActorHead → action_mean = Σ αᵢ · candidate_dᵢ + residual (B, N_VARS+1)
        └── ValueHead → V(s) (B, 1)
"""

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces
from typing import Dict, Tuple

from bundle_norm_RL.script1.features_extractor import IncrementalLevelBundleFeaturesExtractor


def _orthogonal_init(module, gain=1.0):
    """对 Linear 层做正交初始化"""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class ActorHead(nn.Module):
    """Actor Head: 输出 action_mean (B, N_VARS+1)"""
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.net[:-1].apply(lambda m: _orthogonal_init(m, gain=np.sqrt(2)))
        _orthogonal_init(self.net[-1], gain=0.01)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class CutAwareActorHead(nn.Module):
    """
    Cut-Aware Actor Head: 基于 KKT 凸组合的 action_mean 计算（增量版本）

    action_mean = Σᵢ αᵢ · candidate_dᵢ + MLP_residual([seq_emb; global_emb])

    其中 candidate_dᵢ 是每个 cut 的候选增量
    """
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.action_dim = action_dim

        self.residual_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.residual_net[:-1].apply(lambda m: _orthogonal_init(m, gain=np.sqrt(2)))
        _orthogonal_init(self.residual_net[-1], gain=0.01)

    def forward(
        self,
        h: torch.Tensor,
        candidate_actions: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        convex_comb = torch.einsum('bk,bkd->bd', attention_weights, candidate_actions)
        residual = self.residual_net(h)
        return convex_comb + residual


class ValueHead(nn.Module):
    """Value Head: 输出 V(s)"""
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
        for i in range(0, len(self.net) - 1, 3):
            _orthogonal_init(self.net[i], gain=np.sqrt(2))
        _orthogonal_init(self.net[-1], gain=1.0)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class IncrementalLevelBundleActorCriticPolicy(ActorCriticPolicy):
    """
    自定义 Actor-Critic 策略（增量版本）

    继承 SB3 的 ActorCriticPolicy，复用 action_dist 和 log_std，
    重写 _build/forward/evaluate_actions/predict_values/_predict。
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        lr_schedule,
        features_extractor_class=IncrementalLevelBundleFeaturesExtractor,
        features_extractor_kwargs=None,
        **kwargs,
    ):
        if features_extractor_kwargs is None:
            features_extractor_kwargs = dict(hidden_dim=128)

        if "net_arch" not in kwargs:
            kwargs["net_arch"] = []

        self._encoder_type = features_extractor_kwargs.get("encoder_type", "deepset")

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            **kwargs,
        )

    def _build(self, lr_schedule):
        """重写 _build: 跳过 SB3 内置的 mlp_extractor/action_net/value_net"""
        hidden_dim = self.features_extractor.hidden_dim
        combined_dim = 2 * hidden_dim
        action_dim = self.action_space.shape[0]

        if self._encoder_type == "cut_aware":
            self.actor_head = CutAwareActorHead(combined_dim, action_dim, hidden_dim)
        else:
            self.actor_head = ActorHead(combined_dim, action_dim, hidden_dim)

        self.value_head = ValueHead(combined_dim, hidden_dim)

        self.log_std = nn.Parameter(
            torch.ones(action_dim) * -3.0,
            requires_grad=True,
        )

        from stable_baselines3.common.torch_layers import MlpExtractor

        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=[],
            activation_fn=self.activation_fn,
            device=self.device,
        )
        self.action_net = nn.Linear(combined_dim, action_dim)
        self.value_net = nn.Linear(combined_dim, 1)

        self.mlp_extractor.latent_dim_pi = combined_dim
        self.mlp_extractor.latent_dim_vf = combined_dim

        active_params = list(self.actor_head.parameters()) + \
                        list(self.value_head.parameters()) + \
                        list(self.features_extractor.parameters()) + \
                        [self.log_std]

        self.optimizer = self.optimizer_class(
            active_params,
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _encode(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.features_extractor(obs)

    def _compute_action_mean(self, h: torch.Tensor) -> torch.Tensor:
        if self._encoder_type == "cut_aware":
            candidate_actions = self.features_extractor.candidate_actions
            attention_weights = self.features_extractor.attention_weights
            if candidate_actions is None or attention_weights is None:
                return ActorHead(h.shape[-1], self.action_space.shape[0], h.shape[-1] // 2).to(h.device)(h)
            return self.actor_head(h, candidate_actions, attention_weights)
        else:
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
