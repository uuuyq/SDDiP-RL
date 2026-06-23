"""
Conv1d Encoder for Level Bundle RL

使用 1D 卷积编码次梯度历史序列，结合全局特征编码器。

输入:
├── subgradient_history: (B, K, N_VARS+1) - 次梯度历史
├── valid_mask: (B, K) - 有效掩码
├── pi: (B, N_VARS) - 当前乘子
├── pi0: (B, 1) - 当前 pi0
├── lb_ub: (B, 3) - [LB, UB, gap]
├── trial_point: (B, trial_point_dim)
└── realization: (B, realization_dim)

输出:
├── sequence_embedding: (B, hidden_dim) - Conv1d 编码的序列表示
└── global_embedding: (B, hidden_dim) - 全局信息编码
"""

import torch
import torch.nn as nn


class Conv1dSubgradientEncoder(nn.Module):
    """
    使用 1D 卷积编码次梯度历史序列

    输入: (B, K, N_VARS+1) → 转置为 (B, N_VARS+1, K) → Conv1d → (B, hidden_dim)
    """

    def __init__(self, state_dim: int, K: int, hidden_dim: int = 64):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(state_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # 使用 adaptive pooling 将变长序列压缩为固定长度
        self.pool = nn.AdaptiveAvgPool1d(1)

        # 将 masked 位置置零后 pooling
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, subgradient_history: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            subgradient_history: (B, K, state_dim)
            valid_mask: (B, K), 1=有效, 0=padding

        Returns:
            sequence_embedding: (B, hidden_dim)
        """
        # Conv1d 需要 (B, C, L) 格式
        x = subgradient_history.transpose(1, 2)  # (B, state_dim, K)

        # mask: 将无效位置置零
        mask = valid_mask.unsqueeze(1)  # (B, 1, K)
        x = x * mask

        x = self.conv(x)  # (B, hidden_dim, K)
        x = x * mask  # 再次 mask（卷积可能泄漏 padding 信息）

        x = self.pool(x).squeeze(-1)  # (B, hidden_dim)
        x = self.proj(x)

        return x


class GlobalEncoder(nn.Module):
    """
    全局信息编码器

    输入: [pi, pi0, lb_ub, trial_point, realization]
    输出: global_embedding (hidden_dim)
    """

    def __init__(
        self,
        n_vars: int,
        trial_point_dim: int,
        realization_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()

        # pi(N_VARS) + pi0(1) + lb_ub(3) + trial_point + realization
        global_input_dim = n_vars + 1 + 3 + trial_point_dim + realization_dim

        self.encoder = nn.Sequential(
            nn.LayerNorm(global_input_dim),
            nn.Linear(global_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        pi: torch.Tensor,
        pi0: torch.Tensor,
        lb_ub: torch.Tensor,
        trial_point: torch.Tensor,
        realization: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            global_embedding: (B, hidden_dim)
        """
        x = torch.cat([pi, pi0, lb_ub, trial_point, realization], dim=-1)
        return self.encoder(x)


class LevelBundleEncoder(nn.Module):
    """
    完整的 Level Bundle 编码器

    整合 Conv1dSubgradientEncoder + GlobalEncoder

    输出:
        sequence_embedding: (B, hidden_dim) - 次梯度序列编码
        global_embedding: (B, hidden_dim) - 全局信息编码
    """

    def __init__(
        self,
        state_dim: int,
        n_vars: int,
        trial_point_dim: int,
        realization_dim: int,
        K: int,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.conv_encoder = Conv1dSubgradientEncoder(state_dim, K, hidden_dim)
        self.global_encoder = GlobalEncoder(
            n_vars, trial_point_dim, realization_dim, hidden_dim
        )

    def forward(
        self,
        subgradient_history: torch.Tensor,
        valid_mask: torch.Tensor,
        pi: torch.Tensor,
        pi0: torch.Tensor,
        lb_ub: torch.Tensor,
        trial_point: torch.Tensor,
        realization: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            sequence_embedding: (B, hidden_dim)
            global_embedding: (B, hidden_dim)
        """
        seq_emb = self.conv_encoder(subgradient_history, valid_mask)
        global_emb = self.global_encoder(pi, pi0, lb_ub, trial_point, realization)
        return seq_emb, global_emb
