"""
Encoder for Level Bundle RL

支持三种次梯度集合编码器:
1. DeepSetEncoder: 置换不变的集合编码 (ρ( Σ_i φ(x_i) ))
2. CrossAttentionEncoder: Cross-Attention 编码，用全局状态作为 query 关注最相关的 cut
3. SelfAttentionEncoder: Self-Attention 编码，cuts 之间相互关注捕捉交互关系

通过 encoder_type 参数切换，默认 "deepset"。

输入:
├── subgradient_history: (B, K, 2*(N_VARS+1)) - 次梯度集合 + 生成乘子
│   每行 = [subgradient_pi(N_VARS), subgradient_pi0(1), gen_pi(N_VARS), gen_pi0(1)]
├── valid_mask: (B, K) - 有效掩码
├── pi: (B, N_VARS) - 当前乘子
├── pi0: (B, 1) - 当前 pi0
├── lb_ub: (B, 3) - [LB, UB, gap]
├── trial_point: (B, trial_point_dim)
└── realization: (B, realization_dim)

输出:
├── set_embedding: (B, hidden_dim) - 次梯度集合编码
└── global_embedding: (B, hidden_dim) - 全局信息编码
"""

import torch
import torch.nn as nn


class DeepSetEncoder(nn.Module):
    """
    DeepSet 编码器: ρ( Σ_i φ(x_i) )

    输入: (B, K, state_dim) → φ → masked sum → ρ → (B, hidden_dim)
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()

        # φ: 逐元素变换
        self.phi = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # ρ: 聚合后变换
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, subgradient_history: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            subgradient_history: (B, K, state_dim)
            valid_mask: (B, K), 1=有效, 0=padding

        Returns:
            set_embedding: (B, hidden_dim)
        """
        # φ: 逐元素变换
        h = self.phi(subgradient_history)  # (B, K, hidden_dim)

        # mask: 将无效位置置零
        h = h * valid_mask.unsqueeze(-1)  # (B, K, hidden_dim)

        # 聚合: masked sum (置换不变)
        h = h.sum(dim=1)  # (B, hidden_dim)

        # ρ: 聚合后变换
        h = self.rho(h)  # (B, hidden_dim)

        return h


class TransformerBlock(nn.Module):
    """
    标准 Transformer block: MultiheadSelfAttention + FFN + LayerNorm + Residual
    """
    def __init__(self, hidden_dim: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class CrossAttentionEncoder(nn.Module):
    """
    Cross-Attention 编码器: 用全局状态作为 query，次梯度 cuts 作为 key/value

    让网络根据当前乘子位置动态关注最相关的 cut。
    """

    def __init__(
        self,
        state_dim: int,
        n_vars: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # 将 subgradient cut 投影到 hidden_dim
        self.cut_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 将全局状态 (pi + pi0 + lb_ub) 投影为 query
        self.query_proj = nn.Sequential(
            nn.Linear(n_vars + 1 + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Cross-Attention 层（独立实现，不使用 TransformerBlock）
        self.attn_layers = nn.ModuleList()
        self.norm1_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm2_layers = nn.ModuleList()

        for _ in range(n_layers):
            self.attn_layers.append(nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=n_heads, batch_first=True,
            ))
            self.norm1_layers.append(nn.LayerNorm(hidden_dim))
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            ))
            self.norm2_layers.append(nn.LayerNorm(hidden_dim))

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        subgradient_history: torch.Tensor,
        valid_mask: torch.Tensor,
        pi: torch.Tensor,
        pi0: torch.Tensor,
        lb_ub: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            set_embedding: (B, hidden_dim)
        """
        # 构建 query: (B, 1, hidden_dim)
        global_state = torch.cat([pi, pi0, lb_ub], dim=-1)
        query = self.query_proj(global_state).unsqueeze(1)

        # 构建 key/value: (B, K, hidden_dim)
        kv = self.cut_proj(subgradient_history)

        # attention mask: True = 忽略的位置
        attn_mask = (valid_mask == 0)

        # Cross-Attention 层: query attend to kv
        x = query  # (B, 1, hidden_dim)
        for attn, norm1, ffn, norm2 in zip(
            self.attn_layers, self.norm1_layers,
            self.ffn_layers, self.norm2_layers,
        ):
            attn_out, _ = attn(query=x, key=kv, value=kv, key_padding_mask=attn_mask)
            x = norm1(x + attn_out)
            ffn_out = ffn(x)
            x = norm2(x + ffn_out)

        x = x.squeeze(1)
        x = self.output_proj(x)

        return x


class SelfAttentionEncoder(nn.Module):
    """
    Self-Attention 编码器: cuts 之间相互关注，捕捉 cut 间的交互关系

    与 CrossAttention 的区别:
    - 不使用 global state 作为 query，cuts 自己关注自己
    - 通过 [CLS] token (learnable) 聚合所有 cut 的信息
    - Global info 独立编码，最终在 actor/critic 中与 set_embedding 拼接

    结构:
    - cut_proj: 将每个 cut 映射到 hidden_dim
    - [CLS] token: 可学习的聚合 token，与 cuts 一起做 self-attention
    - 多层 Transformer block (Self-Attention + FFN)
    - 输出 [CLS] 位置的 embedding 作为 set_embedding
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # 将 subgradient cut 投影到 hidden_dim
        self.cut_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # [CLS] token: 可学习的聚合 token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Self-Attention 层（标准 Transformer blocks）
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads) for _ in range(n_layers)
        ])

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        subgradient_history: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            subgradient_history: (B, K, state_dim)
            valid_mask: (B, K), 1=有效, 0=padding

        Returns:
            set_embedding: (B, hidden_dim)
        """
        B = subgradient_history.shape[0]
        K = subgradient_history.shape[1]

        # 投影 cuts: (B, K, hidden_dim)
        cut_emb = self.cut_proj(subgradient_history)

        # 拼接 [CLS] token: (B, K+1, hidden_dim)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, cut_emb], dim=1)  # (B, K+1, hidden_dim)

        # 构建 attention mask: 对 [CLS] 位置不 mask，cuts 按 valid_mask 处理
        # key_padding_mask: (B, K+1), True = 忽略
        # 在 cls 位置(索引0) 补 False
        cls_mask = torch.zeros(B, 1, dtype=valid_mask.dtype, device=valid_mask.device)
        attn_mask = torch.cat([cls_mask, (valid_mask == 0)], dim=1)  # (B, K+1)

        # Self-Attention 层
        for block in self.blocks:
            x = block(x, key_padding_mask=attn_mask)

        # 取 [CLS] 位置的输出: (B, hidden_dim)
        cls_out = x[:, 0, :]

        # 输出投影
        cls_out = self.output_proj(cls_out)

        return cls_out


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

    支持三种次梯度集合编码器:
    - "deepset": DeepSetEncoder (置换不变，sum pooling)
    - "cross_attention": CrossAttentionEncoder (global→cut 交叉关注)
    - "self_attention": SelfAttentionEncoder (cuts 之间相互关注，[CLS] 聚合)

    输出:
        set_embedding: (B, hidden_dim) - 次梯度集合编码
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
        encoder_type: str = "deepset",
        n_heads: int = 4,
        n_attn_layers: int = 2,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type

        if encoder_type == "deepset":
            self.set_encoder = DeepSetEncoder(state_dim, hidden_dim)
        elif encoder_type == "cross_attention":
            self.set_encoder = CrossAttentionEncoder(
                state_dim=state_dim,
                n_vars=n_vars,
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                n_layers=n_attn_layers,
            )
        elif encoder_type == "self_attention":
            self.set_encoder = SelfAttentionEncoder(
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                n_layers=n_attn_layers,
            )
        else:
            raise ValueError(
                f"Unknown encoder_type: {encoder_type}, "
                f"expected 'deepset', 'cross_attention', or 'self_attention'"
            )

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
            set_embedding: (B, hidden_dim)
            global_embedding: (B, hidden_dim)
        """
        if self.encoder_type == "deepset":
            set_emb = self.set_encoder(subgradient_history, valid_mask)
        elif self.encoder_type == "cross_attention":
            set_emb = self.set_encoder(
                subgradient_history, valid_mask, pi, pi0, lb_ub
            )
        else:  # self_attention
            set_emb = self.set_encoder(subgradient_history, valid_mask)

        global_emb = self.global_encoder(pi, pi0, lb_ub, trial_point, realization)
        return set_emb, global_emb