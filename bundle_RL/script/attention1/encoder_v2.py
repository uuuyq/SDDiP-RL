"""
Attention Encoder V2 - 带输出 Mask

该模块是 encoder.py 的增强版本，在 Attention 输出后对无效位置进行 mask。

改进点：
1. Attention 层使用 key_padding_mask（原有功能）
2. 输出前对 cut_embeddings 进行额外 mask，确保无效位置为零向量
3. 保持与原 encoder.py 的接口兼容

===========================================
              与原版差异
===========================================

原版 encoder.py:
    return cut_embeddings, global_embedding, cls_embedding

新版 encoder_v2.py:
    cut_embeddings = cut_embeddings * valid_mask.unsqueeze(-1)  # 新增
    return cut_embeddings, global_embedding, cls_embedding

这样做的好处：
- 消除 padding 位置的噪声输出
- 保持输入输出的一致性
- 提升训练稳定性
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple


# ============================
# CutEncoder
# ============================

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


# ============================
# SelfAttention
# ============================

class SelfAttention(nn.Module):
    """
    简单的 Self-Attention 层（无位置编码）

    使用 Multi-Head Attention
    """

    def __init__(self, d_model: int = 64, num_heads: int = 4, ffn_dim: int = 128, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: shape (batch_size, seq_len, d_model)
            key_padding_mask: shape (batch_size, seq_len), True 表示需要 mask

        Returns:
            output: shape (batch_size, seq_len, d_model)
        """
        attn_output, _ = self.self_attn(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )

        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x


# ============================
# GlobalEncoder
# ============================

class GlobalEncoder(nn.Module):
    """
    Global Encoder: 编码全局信息

    输入: [pi, trial_point, realization]
    输出: global_embedding (hidden_dim)
    """

    def __init__(self, state_dim: int, trial_point_dim: int, realization_dim: int, hidden_dim: int = 64):
        super().__init__()

        global_input_dim = state_dim + trial_point_dim + realization_dim

        self.encoder = nn.Sequential(
            nn.Linear(global_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(
        self,
        pi: torch.Tensor,
        trial_point: torch.Tensor,
        realization: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pi: shape (batch_size, state_dim)
            trial_point: shape (batch_size, trial_point_dim)
            realization: shape (batch_size, realization_dim)

        Returns:
            global_embedding: shape (batch_size, hidden_dim)
        """
        global_features = torch.cat([pi, trial_point, realization], dim=-1)
        return self.encoder(global_features)


# ============================
# AttentionBundleEncoder (V2)
# ============================

class AttentionBundleEncoder(nn.Module):
    """
    Attention-based Bundle Encoder V2

    整合 CutEncoder、SelfAttention、GlobalEncoder

    与 V1 的区别：
    - 在输出前对 cut_embeddings 进行额外的 mask
    - 确保无效位置的输出为零向量
    """

    def __init__(
        self,
        state_dim: int,
        trial_point_dim: int,
        realization_dim: int,
        K: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        ffn_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()

        self.state_dim = state_dim
        self.trial_point_dim = trial_point_dim
        self.realization_dim = realization_dim
        self.K = K
        self.hidden_dim = hidden_dim

        self.cut_encoder = CutEncoder(state_dim, hidden_dim)

        self.attention_layers = nn.ModuleList([
            SelfAttention(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        self.global_encoder = GlobalEncoder(state_dim, trial_point_dim, realization_dim, hidden_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self,
        cuts: torch.Tensor,
        valid_mask: torch.Tensor,
        pi: torch.Tensor,
        trial_point: torch.Tensor,
        realization: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            cuts: shape (batch_size, K, state_dim)
            valid_mask: shape (batch_size, K), 1 表示有效
            pi: shape (batch_size, state_dim)
            trial_point: shape (batch_size, trial_point_dim)
            realization: shape (batch_size, realization_dim)

        Returns:
            cut_embeddings: shape (batch_size, K, hidden_dim) - 已 mask 无效位置
            global_embedding: shape (batch_size, hidden_dim) - 全局信息编码
            cls_embedding: shape (batch_size, hidden_dim) - 序列级表示
        """
        batch_size = cuts.shape[0]

        # Cut 编码
        cut_features = self._build_cut_features(cuts, pi)
        cut_embeddings = self.cut_encoder(cut_features)

        # 全局编码
        global_embedding = self.global_encoder(pi, trial_point, realization)

        # 构建序列 [CLS, cut_1, cut_2, ..., cut_K]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat([cls_tokens, cut_embeddings], dim=1)

        # 构建 attention mask
        cls_mask = torch.ones(batch_size, 1, device=valid_mask.device)
        extended_mask = torch.cat([cls_mask, valid_mask], dim=1)
        padding_mask = (extended_mask == 0)

        # Self-Attention
        for attention_layer in self.attention_layers:
            sequence = attention_layer(sequence, key_padding_mask=padding_mask)

        # 分离 CLS 和 Cut embeddings
        cls_embedding = sequence[:, 0, :]
        cut_embeddings = sequence[:, 1:, :]

        # ========================================
        # V2 新增：对无效位置进行 mask
        # ========================================
        # 确保 padding 位置的输出为零向量
        cut_embeddings = cut_embeddings * valid_mask.unsqueeze(-1)

        return cut_embeddings, global_embedding, cls_embedding

    def _build_cut_features(self, cuts: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        """
        构建 cut 特征

        Args:
            cuts: shape (batch_size, K, state_dim) - 只有 g
            pi: shape (batch_size, state_dim)

        Returns:
            cut_features: shape (batch_size, K, state_dim)
        """
        return cuts


# ============================
# 便捷函数
# ============================

def create_encoder(
    state_dim: int,
    trial_point_dim: int,
    realization_dim: int,
    K: int,
    hidden_dim: int = 64,
    num_heads: int = 4,
    num_layers: int = 1,
    ffn_dim: int = 128,
    dropout: float = 0.1
) -> AttentionBundleEncoder:
    """
    便捷函数：创建 AttentionBundleEncoder V2

    Returns:
        AttentionBundleEncoder 实例
    """
    return AttentionBundleEncoder(
        state_dim=state_dim,
        trial_point_dim=trial_point_dim,
        realization_dim=realization_dim,
        K=K,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_dim=ffn_dim,
        dropout=dropout
    )
