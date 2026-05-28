"""
DeepSet Encoder 模块

该模块定义了基于 DeepSets 的 Bundle Method 编码器组件，用于从观测中提取特征。

整体网络结构:

输入:
├── cuts: (batch_size, K, state_dim) - K个cut的次梯度
├── valid_mask: (batch_size, K) - 有效cut的掩码
├── pi: (batch_size, state_dim) - 当前对偶点
├── trial_point: (batch_size, trial_point_dim) - 试验点
└── realization: (batch_size, realization_dim) - 场景数据

编码流程:
1. CutEncoder: 对每个cut独立编码
   输入: g_i (state_dim)
   输出: h_i (hidden_dim)

2. Mean Pooling: 使用 valid_mask 计算加权均值
   输入: [h_1, ..., h_K], valid_mask
   输出: h_g (hidden_dim)

3. 全局特征映射: 分别将 pi, trial_point, realization 通过 MLP 映射
   输入: pi, trial_point, realization
   输出: pi_emb, tp_emb, rlz_emb (均为 hidden_dim)

4. 拼接全局特征: z_g = [h_g, pi_emb, tp_emb, rlz_emb]

输出:
├── h: (batch_size, K, hidden_dim) - 各cut的编码
└── z_g: (batch_size, z_g_dim) - 全局特征向量

核心设计决策:
- DeepSet 使用 Sum/Mean Pooling 处理集合数据
- 使用 MLP 映射后再拼接，避免直接拼接原始特征
- 集合操作顺序无关，计算复杂度为 O(K)
"""
import torch
import torch.nn as nn
from typing import Tuple


class CutEncoder(nn.Module):
    """
    Cut Encoder: 对每个 cut 进行独立编码

    输入: g_i (subgradient, shape (state_dim,))
    输出: h_i, shape (hidden_dim,)
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, cut_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cut_features: shape (batch_size, K, state_dim)

        Returns:
            embeddings: shape (batch_size, K, hidden_dim)
        """
        return self.encoder(cut_features)


class GlobalFeatureNetwork(nn.Module):
    """
    Global Feature Network: 将各个全局特征映射到统一维度

    输入: pi, trial_point, realization
    输出: 映射后的特征（均为 hidden_dim）
    """

    def __init__(self, state_dim: int, trial_point_dim: int, realization_dim: int, hidden_dim: int = 64):
        super().__init__()

        # Pi 映射网络
        self.pi_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Trial Point 映射网络
        self.tp_net = nn.Sequential(
            nn.Linear(trial_point_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Realization 映射网络
        self.rlz_net = nn.Sequential(
            nn.Linear(realization_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, pi: torch.Tensor, trial_point: torch.Tensor, realization: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pi: shape (batch_size, state_dim)
            trial_point: shape (batch_size, trial_point_dim)
            realization: shape (batch_size, realization_dim)

        Returns:
            pi_emb: shape (batch_size, hidden_dim)
            tp_emb: shape (batch_size, hidden_dim)
            rlz_emb: shape (batch_size, hidden_dim)
        """
        pi_emb = self.pi_net(pi)
        tp_emb = self.tp_net(trial_point)
        rlz_emb = self.rlz_net(realization)
        return pi_emb, tp_emb, rlz_emb


class DeepSetFeatureExtractor(nn.Module):
    """
    DeepSet Feature Extractor

    整合 CutEncoder、MeanPooling、GlobalFeatureNetwork

    核心特点:
    - 使用 MLP 映射后再拼接，避免直接拼接原始特征
    - O(K) 线性复杂度
    - 顺序无关的集合操作
    """

    def __init__(
        self,
        state_dim: int,
        trial_point_dim: int,
        realization_dim: int,
        K: int,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        self.state_dim = state_dim
        self.trial_point_dim = trial_point_dim
        self.realization_dim = realization_dim
        self.K = K
        self.hidden_dim = hidden_dim

        self.cut_encoder = CutEncoder(state_dim, hidden_dim)
        self.global_net = GlobalFeatureNetwork(state_dim, trial_point_dim, realization_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        cuts: torch.Tensor,
        valid_mask: torch.Tensor,
        pi: torch.Tensor,
        trial_point: torch.Tensor,
        realization: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cuts: shape (batch_size, K, state_dim)
            valid_mask: shape (batch_size, K), 1 表示有效
            pi: shape (batch_size, state_dim)
            trial_point: shape (batch_size, trial_point_dim)
            realization: shape (batch_size, realization_dim)

        Returns:
            h: shape (batch_size, K, hidden_dim) - 各cut的编码
            z_g: shape (batch_size, z_g_dim) - 全局特征向量
        """
        batch_size = cuts.shape[0]

        # Step 1: Cut Encoding
        cut_features = cuts  # 仅使用 g_i
        h = self.cut_encoder(cut_features)  # (batch_size, K, hidden_dim)

        # Step 2: Mean Pooling（必须使用 mask）
        # CutEncoder 的输出不再是零，必须显式遮蔽无效位置
        valid_mask_expanded = valid_mask.unsqueeze(-1).float()  # (batch_size, K, 1)
        masked_h = h * valid_mask_expanded
        sum_h = masked_h.sum(dim=1)  # (batch_size, hidden_dim)
        count = valid_mask.sum(dim=1, keepdim=True).float() + 1e-8  # (batch_size, 1)
        h_g = sum_h / count  # (batch_size, hidden_dim)

        # Step 3: 全局特征映射（分别 MLP 后再拼接）
        pi_emb, tp_emb, rlz_emb = self.global_net(pi, trial_point, realization)

        # Step 4: 拼接全局特征
        z_g = torch.cat([h_g, pi_emb, tp_emb, rlz_emb], dim=-1)  # (batch_size, 4 * hidden_dim)
        z_g = self.dropout(z_g)

        return h, z_g