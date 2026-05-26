
"""
Cut Encoder 模块
编码单个 cut 的特征
"""
import torch
import torch.nn as nn


class CutEncoder(nn.Module):
    """
    Cut Encoder: 对每个 cut 进行独立编码

    输入: cut 特征 (g_i, phi_i, ...)
    输出: cut embedding
    """

    def __init__(self, cut_dim: int, hidden_dim: int = 64):
        """
        Args:
            cut_dim: 单个 cut 的特征维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()

        self.cut_dim = cut_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(cut_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, cut_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cut_features: shape (batch_size, num_cuts, cut_dim)

        Returns:
            embeddings: shape (batch_size, num_cuts, hidden_dim)
        """
        return self.encoder(cut_features)
