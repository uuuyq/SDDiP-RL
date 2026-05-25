"""
Attention Encoder 模块

该模块定义了 Attention-based Bundle Method 的编码器组件，用于从观测中提取特征。

整体网络结构:

输入:
├── cuts: (batch_size, K, state_dim) - K个cut的次梯度
├── valid_mask: (batch_size, K) - 有效cut的掩码
├── pi: (batch_size, state_dim) - 当前对偶点
├── trial_point: (batch_size, trial_point_dim) - 试验点
└── realization: (batch_size, realization_dim) - 场景数据

编码流程:
1. CutEncoder: 对每个cut独立编码
   输入: [g_i, phi_i, pi_i] (state_dim + 1 + state_dim)
   输出: h_i (hidden_dim)

2. SelfAttention (多层): 对cut序列进行自注意力
   输入: [cls_token, h_1, ..., h_K]
   输出: [cls_embedding, h'_1, ..., h'_K]

3. GlobalEncoder: 编码全局信息
   输入: [pi, trial_point, realization]
   输出: global_embedding (hidden_dim)

输出:
├── cut_embeddings: (batch_size, K, hidden_dim) - 编码后的cut序列
├── global_embedding: (batch_size, hidden_dim) - 全局信息编码
└── cls_embedding: (batch_size, hidden_dim) - 序列级表示

组件说明:
- CutEncoder: 单个cut的编码器
- SelfAttention: 多头自注意力层
- GlobalEncoder: 全局信息编码器
- AttentionBundleEncoder: 完整的编码器（整合上述组件）
"""
import torch
import torch.nn as nn


class CutEncoder(nn.Module):
    """
    Cut Encoder: 对每个 cut 进行独立编码
    
    输入: cut_feature = [g_i, phi_i, pi_i]
        - g_i: subgradient, shape (state_dim,)
        - phi_i: cut 对应目标值, scalar
        - pi_i: 生成该 cut 时的 dual point, shape (state_dim,)
    
    输出: h_i, shape (hidden_dim,)
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        input_dim = state_dim + 1 + state_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, cut_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cut_features: shape (batch_size, K, input_dim)
        
        Returns:
            embeddings: shape (batch_size, K, hidden_dim)
        """
        return self.encoder(cut_features)


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
            x: shape (batch_size, K, d_model)
            key_padding_mask: shape (batch_size, K), True 表示无效位置
        
        Returns:
            output: shape (batch_size, K, d_model)
        """
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x


class GlobalEncoder(nn.Module):
    """
    Global Encoder: 编码全局信息
    
    输入: [pi, trial_point, realization]
    输出: global_embedding, shape (hidden_dim,)
    """
    
    def __init__(self, state_dim: int, trial_point_dim: int, realization_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        input_dim = state_dim + trial_point_dim + realization_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, pi: torch.Tensor, trial_point: torch.Tensor, realization: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pi: shape (batch_size, state_dim)
            trial_point: shape (batch_size, trial_point_dim)
            realization: shape (batch_size, realization_dim)
        
        Returns:
            global_embedding: shape (batch_size, hidden_dim)
        """
        x = torch.cat([pi, trial_point, realization], dim=-1)
        return self.encoder(x)


class AttentionBundleEncoder(nn.Module):
    """
    Attention Bundle Encoder: 完整的编码器
    
    整合 CutEncoder, SelfAttention, GlobalEncoder
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            cuts: shape (batch_size, K, state_dim)
            valid_mask: shape (batch_size, K), 1 表示有效
            pi: shape (batch_size, state_dim)
            trial_point: shape (batch_size, trial_point_dim)
            realization: shape (batch_size, realization_dim)
        
        Returns:
            cut_embeddings: shape (batch_size, K, hidden_dim)
            global_embedding: shape (batch_size, hidden_dim)
            cls_embedding: shape (batch_size, hidden_dim)
        """
        batch_size = cuts.shape[0]
        
        cut_features = self._build_cut_features(cuts, pi)
        cut_embeddings = self.cut_encoder(cut_features)
        
        global_embedding = self.global_encoder(pi, trial_point, realization)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat([cls_tokens, cut_embeddings], dim=1)
        
        cls_mask = torch.ones(batch_size, 1, device=valid_mask.device)
        extended_mask = torch.cat([cls_mask, valid_mask], dim=1)
        
        padding_mask = (extended_mask == 0)
        
        for attention_layer in self.attention_layers:
            sequence = attention_layer(sequence, key_padding_mask=padding_mask)
        
        cls_embedding = sequence[:, 0, :]
        cut_embeddings = sequence[:, 1:, :]
        
        return cut_embeddings, global_embedding, cls_embedding
    
    def _build_cut_features(self, cuts: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        # TODO 后面可以进一步完善cuts的特征，当前只使用 g
        """
        构建 cut 特征: [g_i, phi_i, pi_i]
        
        由于当前环境只提供 g，phi 和 pi 需要从其他信息推断或使用默认值
        这里简化为: [g_i, phi_approx, pi_current]
        
        Args:
            cuts: shape (batch_size, K, state_dim) - 只有 g
            pi: shape (batch_size, state_dim)
        
        Returns:
            cut_features: shape (batch_size, K, state_dim + 1 + state_dim)
        """
        # batch_size, K, state_dim = cuts.shape
        #
        # phi_approx = torch.zeros(batch_size, K, 1, device=cuts.device)
        #
        # pi_expanded = pi.unsqueeze(1).expand(-1, K, -1)
        #
        # cut_features = torch.cat([cuts, phi_approx, pi_expanded], dim=-1)
        
        return cuts