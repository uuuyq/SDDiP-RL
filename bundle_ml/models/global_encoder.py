
"""
Global Encoder 模块
编码全局特征 (λ, x_prev, realization, stage)
"""
import torch
import torch.nn as nn


class GlobalEncoder(nn.Module):
    """
    Global Encoder: 编码全局特征

    输入: [lambda_, x_prev, realization, stage]
    输出: global embedding
    """

    def __init__(
        self,
        lambda_dim: int,
        x_prev_dim: int,
        realization_dim: int,
        hidden_dim: int = 64,
        use_stage_embedding: bool = True,
        max_stages: int = 100,
    ):
        """
        Args:
            lambda_dim: λ 的维度
            x_prev_dim: x_prev 的维度
            realization_dim: realization 的维度
            hidden_dim: 隐藏层维度
            use_stage_embedding: 是否使用 stage embedding
            max_stages: 最大 stage 数量
        """
        super().__init__()

        self.lambda_dim = lambda_dim
        self.x_prev_dim = x_prev_dim
        self.realization_dim = realization_dim
        self.hidden_dim = hidden_dim
        self.use_stage_embedding = use_stage_embedding

        # Stage embedding
        if use_stage_embedding:
            self.stage_embedding = nn.Embedding(max_stages, hidden_dim)

        # 计算全局输入维度
        global_input_dim = lambda_dim + x_prev_dim + realization_dim

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(global_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 如果使用 stage embedding，需要额外的融合层
        if use_stage_embedding:
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def forward(
        self,
        lambda_: torch.Tensor,
        x_prev: torch.Tensor,
        realization: torch.Tensor,
        stage: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            lambda_: shape (batch_size, lambda_dim)
            x_prev: shape (batch_size, x_prev_dim)
            realization: shape (batch_size, realization_dim)
            stage: shape (batch_size,) or None, stage 索引

        Returns:
            global_embedding: shape (batch_size, hidden_dim)
        """
        # 拼接全局特征
        global_features = torch.cat([lambda_, x_prev, realization], dim=-1)

        # 编码
        embedding = self.encoder(global_features)

        # 融合 stage embedding
        if self.use_stage_embedding and stage is not None:
            stage_emb = self.stage_embedding(stage.long())
            embedding = self.fusion(torch.cat([embedding, stage_emb], dim=-1))

        return embedding
