"""
Features Extractor 模块

该模块定义了与 stable-baselines3 兼容的特征提取器，是当前训练流程的核心组件。

===========================================
              整体网络架构
===========================================

当前训练流程使用 stable-baselines3 的 `MultiInputActorCriticPolicy`：
┌─────────────────────────────────────────────────────────────────┐
│                    MultiInputActorCriticPolicy                  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           AttentionFeaturesExtractor (共享)             │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │        AttentionBundleEncoder                   │   │   │
│  │  │  ┌─────────┐  ┌───────────────┐  ┌──────────┐   │   │   │
│  │  │  │CutEncoder│→│Self-Attention │→│GlobalEnc │   │   │   │
│  │  │  └─────────┘  │   (with CLS)  │  └──────────┘   │   │   │
│  │  │               └───────────────┘                  │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │                           ↓                             │   │
│  │              ┌─────────────────────────┐               │   │
│  │              │ 特征聚合 (拼接)         │               │   │
│  │              │ pool + global + cls     │               │   │
│  │              └───────────┬─────────────┘               │   │
│  │                          ↓                             │   │
│  │              ┌─────────────────────────┐               │   │
│  │              │   MLP 特征变换          │               │   │
│  │              └───────────┬─────────────┘               │   │
│  └───────────────────────────┼───────────────────────────┘   │
│                              ↓                               │
│         ┌────────────────────┴────────────────────┐          │
│         ↓                                         ↓          │
│  ┌──────────────┐                        ┌──────────────┐     │
│  │   Actor MLP  │                        │  Critic MLP  │     │
│  │  [128, 128]  │                        │  [128, 128]  │     │
│  └──────┬───────┘                        └──────┬───────┘     │
│         ↓                                       ↓             │
│  ┌──────────────┐                        ┌──────────────┐     │
│  │  action_net  │                        │  value_net   │     │
│  │ (输出均值)   │                        │  (输出V(s))  │     │
│  └──────────────┘                        └──────────────┘     │
└─────────────────────────────────────────────────────────────────┘

===========================================
              组件职责说明
===========================================

1. AttentionFeaturesExtractor:
   - 实现 stable_baselines3 的 BaseFeaturesExtractor 接口
   - 从 Dict 类型的 observation 中提取特征
   - **Actor 和 Critic 共享同一个实例**（由 stable-baselines3 自动处理）

2. AttentionBundleEncoder:
   - 核心编码器，包含 CutEncoder、Self-Attention、GlobalEncoder
   - 输出三种特征: pool_embedding、global_embedding、cls_embedding

3. MultiInputActorCriticPolicy (stable-baselines3):
   - 自动将 features_extractor 的输出分别传入 Actor 和 Critic 的 MLP
   - 确保特征提取器的参数在 Actor 和 Critic 之间共享

===========================================
              输入输出规格
===========================================

输入 (observation_space.Dict):
├── cuts: (batch_size, K, state_dim)      - K个cut的次梯度
├── valid_mask: (batch_size, K)           - 有效cut的掩码
├── pi: (batch_size, state_dim)           - 当前对偶点
├── trial_point: (batch_size, trial_point_dim) - 试验点
└── realization: (batch_size, realization_dim) - 场景数据

输出 (供 Actor/Critic 使用):
└── features: (batch_size, features_dim)  - 提取的特征向量

===========================================
              使用方式
===========================================

在 train.py 中:
    policy_kwargs = dict(
        features_extractor_class=AttentionFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )
    model = PPO(
        policy=MultiInputActorCriticPolicy,
        env=env,
        policy_kwargs=policy_kwargs
    )

这样配置后，stable-baselines3 会自动：
1. 创建一个共享的 AttentionFeaturesExtractor 实例
2. Actor 和 Critic 的 MLP 层独立，但共享特征提取器的参数
3. 反向传播时同时更新 Actor 和 Critic 的参数
"""
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from typing import Dict

from bundle_RL.script.attention.encoder import AttentionBundleEncoder


class AttentionFeaturesExtractor(BaseFeaturesExtractor):
    """
    Attention-based Features Extractor for stable_baselines3
    
    从 observation 中提取特征，供 Actor 和 Critic 使用。
    该特征提取器会被 Actor 和 Critic 共享。
    
    observation_space 结构:
        - cuts: (K, state_dim)
        - valid_mask: (K,)
        - pi: (state_dim,)
        - trial_point: (trial_point_dim,)
        - realization: (realization_dim,)
    """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 128,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        ffn_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__(observation_space, features_dim)
        
        self.cuts_shape = observation_space["cuts"].shape
        self.valid_mask_shape = observation_space["valid_mask"].shape
        self.pi_shape = observation_space["pi"].shape
        self.trial_point_shape = observation_space["trial_point"].shape
        self.realization_shape = observation_space["realization"].shape
        
        self.K = self.cuts_shape[0]
        self.state_dim = self.cuts_shape[1]
        self.trial_point_dim = self.trial_point_shape[0]
        self.realization_dim = self.realization_shape[0]
        
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
        
        combined_dim = hidden_dim * 3  # pool_embedding + global_embedding + cls_embedding
        
        self.net = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        cuts = observations["cuts"]
        valid_mask = observations["valid_mask"]
        pi = observations["pi"]
        trial_point = observations["trial_point"]
        realization = observations["realization"]
        
        cut_embeddings, global_embedding, cls_embedding = self.encoder(
            cuts, valid_mask, pi, trial_point, realization
        )
        
        valid_mask_expanded = valid_mask.unsqueeze(-1).float()
        masked_embeddings = cut_embeddings * valid_mask_expanded
        pool_embedding = masked_embeddings.sum(dim=1) / (valid_mask.sum(dim=1, keepdim=True) + 1e-8)
        
        combined = torch.cat([pool_embedding, global_embedding, cls_embedding], dim=-1)
        
        features = self.net(combined)
        
        return features