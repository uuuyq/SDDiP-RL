"""
DeepSet Encoder for Level Bundle RL

使用 DeepSet (置换不变网络) 编码次梯度集合，结合全局特征编码器。

DeepSet 原理: ρ( Σ_i φ(x_i) )
- φ: 逐元素变换 (MLP)，将每个 cut 映射到高维空间
- Σ: 置换不变的聚合 (masked sum)
- ρ: 聚合后变换 (MLP)，生成集合表示

输入:
├── subgradient_history: (B, K, N_VARS+1) - 次梯度集合
├── valid_mask: (B, K) - 有效掩码
├── pi: (B, N_VARS) - 当前乘子
├── pi0: (B, 1) - 当前 pi0
├── lb_ub: (B, 3) - [LB, UB, gap]
├── trial_point: (B, trial_point_dim)
└── realization: (B, realization_dim)

输出:
├── set_embedding: (B, hidden_dim) - DeepSet 编码的集合表示
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

    整合 DeepSetEncoder + GlobalEncoder

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
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.deepset_encoder = DeepSetEncoder(state_dim, hidden_dim)
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
        set_emb = self.deepset_encoder(subgradient_history, valid_mask)
        global_emb = self.global_encoder(pi, pi0, lb_ub, trial_point, realization)
        return set_emb, global_emb
