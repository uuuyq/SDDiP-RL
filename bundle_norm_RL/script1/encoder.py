"""
Encoder for Incremental Level Bundle RL

与 script/encoder.py 的区别:
    - GlobalEncoder 额外输入 pi_bar, pi0_bar (稳定中心)
    - CutAwareEncoder 的 candidate_decoder 输出增量 d 而非绝对乘子

支持四种次梯度集合编码器:
1. DeepSetEncoder: 置换不变的集合编码
2. CrossAttentionEncoder: Cross-Attention 编码
3. SelfAttentionEncoder: Self-Attention 编码
4. CutAwareEncoder: 基于 KKT 凸组合的结构化编码（输出增量）

输入:
├── subgradient_history: (B, K, 2*(N_VARS+1))
├── valid_mask: (B, K)
├── pi_bar: (B, N_VARS) - 稳定中心 pi
├── pi0_bar: (B, 1) - 稳定中心 pi0
├── pi: (B, N_VARS) - 当前乘子
├── pi0: (B, 1) - 当前 pi0
├── lb_ub: (B, 3)
├── trial_point: (B, trial_point_dim)
└── realization: (B, realization_dim)

输出:
├── set_embedding: (B, hidden_dim)
└── global_embedding: (B, hidden_dim)
"""

import torch
import torch.nn as nn


class DeepSetEncoder(nn.Module):
    """DeepSet 编码器: ρ( Σ_i φ(x_i) )"""

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, subgradient_history: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        h = self.phi(subgradient_history)
        h = h * valid_mask.unsqueeze(-1)
        h = h.sum(dim=1)
        h = self.rho(h)
        return h


class TransformerBlock(nn.Module):
    """标准 Transformer block"""
    def __init__(self, hidden_dim: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, batch_first=True,
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
    """Cross-Attention 编码器: 用全局状态作为 query，次梯度 cuts 作为 key/value"""

    def __init__(
        self,
        state_dim: int,
        n_vars: int,
        global_state_dim: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.cut_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 全局状态包含 pi_bar, pi0_bar, pi, pi0, lb_ub
        self.query_proj = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

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

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        subgradient_history: torch.Tensor,
        valid_mask: torch.Tensor,
        global_state: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query_proj(global_state).unsqueeze(1)
        kv = self.cut_proj(subgradient_history)
        attn_mask = (valid_mask == 0)

        x = query
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
    """Self-Attention 编码器: cuts 之间相互关注，通过 [CLS] token 聚合"""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.cut_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads) for _ in range(n_layers)
        ])

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
        B = subgradient_history.shape[0]

        cut_emb = self.cut_proj(subgradient_history)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, cut_emb], dim=1)

        cls_mask = torch.zeros(B, 1, dtype=valid_mask.dtype, device=valid_mask.device)
        attn_mask = torch.cat([cls_mask, (valid_mask == 0)], dim=1)

        for block in self.blocks:
            x = block(x, key_padding_mask=attn_mask)

        cls_out = x[:, 0, :]
        cls_out = self.output_proj(cls_out)
        return cls_out


class CutAwareEncoder(nn.Module):
    """
    Cut-Aware 编码器: 基于 KKT 凸组合的结构化编码（增量版本）

    与 script/encoder.py 中 CutAwareEncoder 的区别:
        - candidate_decoder 输出增量 d = (d_pi, d_pi0) 而非绝对乘子
        - 凸组合部分: d = Σᵢ αᵢ · candidate_dᵢ + residual
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
        self.n_vars = n_vars
        self.action_dim = n_vars + 1  # d_pi(N_VARS) + d_pi0(1)

        self.cut_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads) for _ in range(n_layers)
        ])

        # Candidate Decoder: 每个 cut_emb → 候选增量 d
        self.candidate_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim),
        )

        # Weight Generator
        self.weight_generator = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        subgradient_history: torch.Tensor,
        valid_mask: torch.Tensor,
        global_embedding: torch.Tensor,
    ) -> tuple:
        B, K, _ = subgradient_history.shape

        cut_emb = self.cut_proj(subgradient_history)

        attn_mask = (valid_mask == 0)
        for block in self.blocks:
            cut_emb = block(cut_emb, key_padding_mask=attn_mask)

        # Candidate Decoder: 每个 cut → 候选增量
        candidate_actions = self.candidate_decoder(cut_emb)  # (B, K, N_VARS+1)

        # Weight Generator
        global_expanded = global_embedding.unsqueeze(1).expand(B, K, self.hidden_dim)
        weight_input = torch.cat([cut_emb, global_expanded], dim=-1)
        logits = self.weight_generator(weight_input).squeeze(-1)

        logits = logits.masked_fill((valid_mask == 0), float('-inf'))
        attention_weights = torch.softmax(logits, dim=-1)

        weighted_emb = (cut_emb * attention_weights.unsqueeze(-1)).sum(dim=1)
        set_embedding = self.output_proj(weighted_emb)

        return set_embedding, cut_emb, candidate_actions, attention_weights


class GlobalEncoder(nn.Module):
    """
    全局信息编码器（增量版本）

    输入: [pi_bar, pi0_bar, pi, pi0, lb_ub, trial_point, realization]
    额外包含稳定中心 (pi_bar, pi0_bar) 信息
    """

    def __init__(
        self,
        n_vars: int,
        trial_point_dim: int,
        realization_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()

        # pi_bar(N_VARS) + pi0_bar(1) + pi(N_VARS) + pi0(1) + lb_ub(3) + trial_point + realization
        global_input_dim = n_vars + 1 + n_vars + 1 + 3 + trial_point_dim + realization_dim

        self.encoder = nn.Sequential(
            nn.LayerNorm(global_input_dim),
            nn.Linear(global_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        pi_bar: torch.Tensor,
        pi0_bar: torch.Tensor,
        pi: torch.Tensor,
        pi0: torch.Tensor,
        lb_ub: torch.Tensor,
        trial_point: torch.Tensor,
        realization: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([pi_bar, pi0_bar, pi, pi0, lb_ub, trial_point, realization], dim=-1)
        return self.encoder(x)


class IncrementalLevelBundleEncoder(nn.Module):
    """
    完整的增量 Level Bundle 编码器

    支持四种次梯度集合编码器:
    - "deepset": DeepSetEncoder
    - "cross_attention": CrossAttentionEncoder
    - "self_attention": SelfAttentionEncoder
    - "cut_aware": CutAwareEncoder（输出增量 d）

    与 script/encoder.py 的区别:
        - GlobalEncoder 额外输入 pi_bar, pi0_bar
        - CrossAttentionEncoder 使用更完整的全局状态
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

        self._cut_embeddings = None
        self._candidate_actions = None
        self._attention_weights = None

        # 全局状态维度: pi_bar + pi0_bar + pi + pi0 + lb_ub
        global_state_dim = n_vars + 1 + n_vars + 1 + 3

        if encoder_type == "deepset":
            self.set_encoder = DeepSetEncoder(state_dim, hidden_dim)
        elif encoder_type == "cross_attention":
            self.set_encoder = CrossAttentionEncoder(
                state_dim=state_dim,
                n_vars=n_vars,
                global_state_dim=global_state_dim,
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
        elif encoder_type == "cut_aware":
            self.set_encoder = CutAwareEncoder(
                state_dim=state_dim,
                n_vars=n_vars,
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                n_layers=n_attn_layers,
            )
        else:
            raise ValueError(
                f"Unknown encoder_type: {encoder_type}, "
                f"expected 'deepset', 'cross_attention', 'self_attention', or 'cut_aware'"
            )

        self.global_encoder = GlobalEncoder(
            n_vars, trial_point_dim, realization_dim, hidden_dim
        )

    @property
    def cut_embeddings(self):
        return self._cut_embeddings

    @property
    def candidate_actions(self):
        return self._candidate_actions

    @property
    def attention_weights(self):
        return self._attention_weights

    def forward(
        self,
        subgradient_history: torch.Tensor,
        valid_mask: torch.Tensor,
        pi_bar: torch.Tensor,
        pi0_bar: torch.Tensor,
        pi: torch.Tensor,
        pi0: torch.Tensor,
        lb_ub: torch.Tensor,
        trial_point: torch.Tensor,
        realization: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 全局编码
        global_emb = self.global_encoder(
            pi_bar, pi0_bar, pi, pi0, lb_ub, trial_point, realization
        )

        if self.encoder_type == "deepset":
            set_emb = self.set_encoder(subgradient_history, valid_mask)
        elif self.encoder_type == "cross_attention":
            global_state = torch.cat([pi_bar, pi0_bar, pi, pi0, lb_ub], dim=-1)
            set_emb = self.set_encoder(subgradient_history, valid_mask, global_state)
        elif self.encoder_type == "self_attention":
            set_emb = self.set_encoder(subgradient_history, valid_mask)
        else:  # cut_aware
            set_emb, cut_embs, cand_acts, attn_w = self.set_encoder(
                subgradient_history, valid_mask, global_emb
            )
            self._cut_embeddings = cut_embs
            self._candidate_actions = cand_acts
            self._attention_weights = attn_w

        return set_emb, global_emb
