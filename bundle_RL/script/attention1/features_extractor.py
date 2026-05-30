"""
Features Extractor 模块（简化版）

该模块定义了与 stable-baselines3 兼容的特征提取器，包装 AttentionBundleEncoder。

===========================================
              整体网络架构
===========================================

attention1 简化版使用自定义的 `AttentionActorCriticPolicy`：

┌─────────────────────────────────────────────────────────────────┐
│                AttentionActorCriticPolicy                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           AttentionFeaturesExtractor (共享)             │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │        AttentionBundleEncoder                   │   │   │
│  │  │  CutEncoder → SelfAttention(+CLS) → GlobalEnc   │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │                           ↓                             │   │
│  │              输出 (B, 2H) = [global ; CLS]              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│         ┌────────────────────┴────────────────────┐              │
│         ↓                                         ↓              │
│  ┌──────────────┐                        ┌──────────────┐        │
│  │ LambdaHead   │                        │  ValueHead   │        │
│  │ Q-K Attention│                        │   MLP        │        │
│  │ (with H)     │                        └──────────────┘        │
│  └──────────────┘                                                │
│  ┌──────────────┐                                                │
│  │  EtaHead     │                                                │
│  │  MLP         │                                                │
│  └──────────────┘                                                │
└─────────────────────────────────────────────────────────────────┘

===========================================
              组件职责说明
===========================================

1. AttentionFeaturesExtractor:
   - 实现 stable_baselines3 的 BaseFeaturesExtractor 接口
   - 持有 AttentionBundleEncoder 实例
   - forward(obs) 返回 [global ; CLS] (B, 2H)，满足 SB3 标准约定
   - encode(obs) 返回完整三元组 (cut_embeddings, global_embedding, cls_embedding)
     供自定义 Policy 调用

2. AttentionBundleEncoder:
   - 核心编码器，输出 (H, global, h_bundle)

3. AttentionActorCriticPolicy (在 policy_network.py 中):
   - 自定义 forward / evaluate_actions / predict_values
   - 直接调用 self.features_extractor.encoder(...) 获取三元组
   - LambdaHead 利用 H 做 Q-K 打分，EtaHead/ValueHead 仅用 [global;CLS]

===========================================
              输入输出规格
===========================================

输入 (observation_space.Dict):
├── cuts: (batch_size, K, state_dim)
├── valid_mask: (batch_size, K)
├── pi: (batch_size, state_dim)
├── trial_point: (batch_size, trial_point_dim)
└── realization: (batch_size, realization_dim)

forward 输出:
└── features: (batch_size, 2 * hidden_dim)  - [global ; CLS]

encode 输出:
├── cut_embeddings: (batch_size, K, hidden_dim)
├── global_embedding: (batch_size, hidden_dim)
└── cls_embedding: (batch_size, hidden_dim)
"""
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from typing import Dict, Tuple

from bundle_RL.script.attention1.encoder import AttentionBundleEncoder


class AttentionFeaturesExtractor(BaseFeaturesExtractor):
    """
    Attention-based Features Extractor for stable_baselines3（简化版）

    从 observation 中提取特征，供自定义 Policy 使用。
    该特征提取器会被 Actor 和 Critic 共享。

    observation_space 结构:
        - cuts: (K, state_dim)
        - valid_mask: (K,)
        - pi: (state_dim,)
        - trial_point: (trial_point_dim,)
        - realization: (realization_dim,)

    输出 features_dim = 2 * hidden_dim（[global; CLS]）
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        ffn_dim: int = 128,
        dropout: float = 0.1
    ):
        # features_dim = 2 * hidden_dim（concat of global and cls）
        super().__init__(observation_space, features_dim=2 * hidden_dim)

        self.cuts_shape = observation_space["cuts"].shape
        self.valid_mask_shape = observation_space["valid_mask"].shape
        self.pi_shape = observation_space["pi"].shape
        self.trial_point_shape = observation_space["trial_point"].shape
        self.realization_shape = observation_space["realization"].shape

        self.K = self.cuts_shape[0]
        self.state_dim = self.cuts_shape[1]
        self.trial_point_dim = self.trial_point_shape[0]
        self.realization_dim = self.realization_shape[0]
        self.hidden_dim = hidden_dim

        self.encoder = AttentionBundleEncoder(
            state_dim=self.state_dim,
            trial_point_dim=self.trial_point_dim,
            realization_dim=self.realization_dim,
            K=self.K,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            dropout=dropout
        )

    def encode(
        self,
        observations: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回完整编码三元组，供自定义 Policy 使用

        Returns:
            cut_embeddings: (B, K, H)         - 即 H
            global_embedding: (B, H)
            cls_embedding: (B, H)             - 即 h_bundle
        """
        cuts = observations["cuts"]
        valid_mask = observations["valid_mask"]
        pi = observations["pi"]
        trial_point = observations["trial_point"]
        realization = observations["realization"]

        cut_embeddings, global_embedding, cls_embedding = self.encoder(
            cuts, valid_mask, pi, trial_point, realization
        )
        return cut_embeddings, global_embedding, cls_embedding

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        SB3 标准接口：返回 [global ; CLS] 拼接 (B, 2H)

        在自定义 Policy 中我们通常直接调用 encode()，避免重复计算。
        但保留此接口以满足 SB3 内部约定。
        """
        _, global_embedding, cls_embedding = self.encode(observations)
        return torch.cat([global_embedding, cls_embedding], dim=-1)
