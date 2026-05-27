
"""
Self Attention 模块
对 cut 序列进行自注意力编码
"""
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    简单的 Self-Attention 层（无位置编码）

    使用 Multi-Head Attention
    """

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        ffn_dim: int = 128,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            ffn_dim: FFN 维度
            dropout: Dropout 概率
        """
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: shape (batch_size, seq_len, d_model)
            key_padding_mask: shape (batch_size, seq_len), True 表示需要 mask

        Returns:
            output: shape (batch_size, seq_len, d_model)
        """
        attn_output, _ = self.self_attn(
            x, x, x, key_padding_mask=key_padding_mask, need_weights=False
        )

        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        
        # 在 block 最后重新 mask，防止 padding token 污染 FFN 输出
        if key_padding_mask is not None:
            ffn_output = ffn_output.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        
        x = self.norm2(x + self.dropout(ffn_output))

        return x


class AttentionPooling(nn.Module):
    """
    Attention Pooling 层
    将多个 cut embeddings 聚合成单个 bundle embedding
    """

    def __init__(self, hidden_dim: int = 64):
        """
        Args:
            hidden_dim: 隐藏层维度
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        # Attention 权重网络
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, cut_embeddings: torch.Tensor, valid_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            cut_embeddings: shape (batch_size, num_cuts, hidden_dim)
            valid_mask: shape (batch_size, num_cuts), True 表示有效

        Returns:
            bundle_embedding: shape (batch_size, hidden_dim)
        """
        batch_size = cut_embeddings.shape[0]
        
        # 计算 attention 权重
        attention_scores = self.attention(cut_embeddings).squeeze(-1)  # (batch_size, num_cuts)

        # 应用 mask
        if valid_mask is not None:
            # 将无效的位置设置为 -inf，softmax 后权重为 0
            attention_scores = attention_scores.masked_fill(
                ~valid_mask, torch.finfo(attention_scores.dtype).min
            )

        # 检查是否有样本没有有效的 cuts
        if valid_mask is not None:
            has_valid_cuts = valid_mask.any(dim=-1)  # (batch_size,)
        else:
            has_valid_cuts = torch.ones(batch_size, dtype=torch.bool, device=cut_embeddings.device)
        
        # 对于没有有效 cuts 的样本，将 scores 设置为 0（避免 softmax(-inf) 产生 NaN）
        # 这样 softmax(0, 0, ...) = 均匀分布，乘以零向量 cuts 后结果还是零向量
        attention_scores = torch.where(
            has_valid_cuts.unsqueeze(-1),
            attention_scores,
            torch.zeros_like(attention_scores)
        )
        
        # Softmax 得到权重
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (batch_size, num_cuts)

        # 加权求和
        attention_weights = attention_weights.unsqueeze(-1)  # (batch_size, num_cuts, 1)
        bundle_embedding = torch.sum(cut_embeddings * attention_weights, dim=1)  # (batch_size, hidden_dim)

        return bundle_embedding
