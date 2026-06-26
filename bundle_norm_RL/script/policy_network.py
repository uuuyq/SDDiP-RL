"""
Custom Actor-Critic Policy for Level Bundle RL

直接输出 (pi_raw, pi0_raw)，环境侧做归一化。

网络结构:
    FeaturesExtractor → [seq_emb; global_emb] (B, 2H)
        ├── ActorHead → action_mean (B, N_VARS+1)
        └── ValueHead → V(s) (B, 1) — 更深的网络 + LayerNorm

关键设计:
    - 重写 _build() 以跳过 SB3 内置的 action_net/value_net/mlp_extractor
    - 自定义 ActorHead/ValueHead 做正交初始化（actor 最后一层 gain=0.01）
    - 确保 SB3 内置模块不产生幽灵参数影响优化器
"""

import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces
from typing import Dict, Tuple

from bundle_norm_RL.script.features_extractor import LevelBundleFeaturesExtractor


def _orthogonal_init(module, gain=1.0):
    """对 Linear 层做正交初始化"""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class ActorHead(nn.Module):
    """
    Actor Head: 输出 action_mean (B, N_VARS+1)
    3 层 MLP，足够容量控制 14 维连续动作
    """
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        # 正交初始化: 隐藏层 gain=sqrt(2)，最后一层 gain=0.01（与 SB3 一致）
        self.net[:-1].apply(lambda m: _orthogonal_init(m, gain=np.sqrt(2)))
        _orthogonal_init(self.net[-1], gain=0.01)

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
        # 正交初始化: 隐藏层 gain=sqrt(2)，最后一层 gain=1（与 SB3 value_net 一致）
        for i in range(0, len(self.net) - 1, 3):
            _orthogonal_init(self.net[i], gain=np.sqrt(2))
        _orthogonal_init(self.net[-1], gain=1.0)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


import numpy as np


class LevelBundleActorCriticPolicy(ActorCriticPolicy):
    """
    自定义 Actor-Critic 策略

    继承 SB3 的 ActorCriticPolicy，复用 action_dist 和 log_std，
    重写 _build/forward/evaluate_actions/predict_values/_predict。

    关键: 重写 _build() 以跳过 SB3 内置的 mlp_extractor/action_net/value_net，
    只使用自定义的 ActorHead/ValueHead，避免幽灵参数影响优化器。
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
            features_extractor_kwargs = dict(hidden_dim=128)

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

    def _build(self, lr_schedule):
        """
        重写 _build: 跳过 SB3 内置的 mlp_extractor/action_net/value_net，
        只创建自定义的 ActorHead/ValueHead，并正确设置 log_std。
        """
        # 不调用 super()._build()，完全自定义

        hidden_dim = self.features_extractor.hidden_dim
        combined_dim = 2 * hidden_dim
        action_dim = self.action_space.shape[0]

        # 创建自定义 head
        self.actor_head = ActorHead(combined_dim, action_dim, hidden_dim)
        self.value_head = ValueHead(combined_dim, hidden_dim)

        # log_std: 可学习参数，初始化为较小值
        # log_std=-3 → std≈0.05，14 维空间中避免 KL 过大
        self.log_std = nn.Parameter(
            torch.ones(action_dim) * -3.0,
            requires_grad=True,
        )

        # 创建占位模块以满足 SB3 内部对 mlp_extractor/action_net/value_net 的引用
        # 这些模块不会被 forward 使用，但 SB3 的某些代码可能访问它们的属性
        from stable_baselines3.common.torch_layers import MlpExtractor

        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=[],
            activation_fn=self.activation_fn,
            device=self.device,
        )
        self.action_net = nn.Linear(combined_dim, action_dim)
        self.value_net = nn.Linear(combined_dim, 1)

        # 设置 latent_dim 供 SB3 内部使用
        self.mlp_extractor.latent_dim_pi = combined_dim
        self.mlp_extractor.latent_dim_vf = combined_dim

        # 优化器只包含真正使用的参数
        # 排除占位的 mlp_extractor/action_net/value_net
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
