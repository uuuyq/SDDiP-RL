"""
Features Extractor 模块

该模块定义了与 stable-baselines3 兼容的特征提取器，是当前训练流程的核心组件。

===========================================
              整体网络架构
===========================================

当前训练流程使用自定义策略网络：
┌─────────────────────────────────────────────────────────────────┐
│                    DeepSetPolicyNetwork                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           DeepSetFeaturesExtractor (共享)               │   │
│  │  ┌─────────┐  ┌───────────┐  ┌────────────────────┐    │   │
│  │  │CutEncoder│→│MeanPooling│→│GlobalFeatureNetwork │    │   │
│  │  └─────────┘  └───────────┘  └────────────────────┘    │   │
│  │            ↓                                    ↓       │   │
│  │           h (K×hidden_dim)              z_g (4×hidden_dim) │   │
│  └───────────────────────────┬───────────────────────────┘   │
│                              ↓                               │
│         ┌────────────────────┴────────────────────┐          │
│         ↓                                         ↓          │
│  ┌───────────────────┐                  ┌──────────────┐     │
│  │       Actor       │                  │   Critic     │     │
│  │  ┌─────────────┐  │                  │              │     │
│  │  │QueryNetwork │  │                  │              │     │
│  │  │KeyNetwork   │→│ cut_weights       │              │     │
│  │  │EtaHead      │→│ eta               │              │     │
│  │  └─────────────┘  │                  └──────────────┘     │
│  └───────────────────┘                  ↓                     │
│         ↓                              V(s)                   │
│  ┌──────────────┐                                             │
│  │  action_net  │                                             │
│  │ [cut_weights, eta]                                         │
│  └──────────────┘                                             │
└─────────────────────────────────────────────────────────────────┘

===========================================
              组件职责说明
===========================================

1. DeepSetFeaturesExtractor:
   - 实现 stable_baselines3 的 BaseFeaturesExtractor 接口
   - 从 Dict 类型的 observation 中提取特征
   - 输出 h (各cut编码) 和 z_g (全局特征)

2. Actor:
   - 使用 QueryNetwork 和 KeyNetwork 计算 cut_weights
   - 使用 EtaHead 计算步长 eta
   - 输出 action = [cut_weights, eta]

3. Critic:
   - 使用 z_g 评估状态价值 V(s)

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
├── h: (batch_size, K, hidden_dim)        - 各cut的编码
└── z_g: (batch_size, z_g_dim)            - 全局特征向量
"""
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from typing import Dict, Tuple

from bundle_RL.script.deepset.encoder import DeepSetFeatureExtractor as DSFE


class DeepSetFeaturesExtractor(BaseFeaturesExtractor):
    """
    DeepSet-based Features Extractor for stable_baselines3
    
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
        
        # 使用新的 DeepSetFeatureExtractor
        self.extractor = DSFE(
            state_dim=self.state_dim,
            trial_point_dim=self.trial_point_dim,
            realization_dim=self.realization_dim,
            K=self.K,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # 保存 h 和 z_g 用于后续访问（通过属性方式）
        self._h = None
        self._z_g = None
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        cuts = observations["cuts"]
        valid_mask = observations["valid_mask"]
        pi = observations["pi"]
        trial_point = observations["trial_point"]
        realization = observations["realization"]
        
        # 使用 DeepSetFeatureExtractor 提取特征
        h, z_g = self.extractor(cuts, valid_mask, pi, trial_point, realization)
        
        # 保存供后续使用
        self._h = h
        self._z_g = z_g
        
        # 返回 z_g 作为特征（Critic 使用）
        # Actor 需要同时使用 h 和 z_g，会通过属性获取
        return z_g
    
    @property
    def h(self) -> torch.Tensor:
        """获取各 cut 的编码"""
        return self._h
    
    @property
    def z_g(self) -> torch.Tensor:
        """获取全局特征向量"""
        return self._z_g