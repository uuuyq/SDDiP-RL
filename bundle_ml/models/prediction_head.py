
"""
Prediction Head 模块
预测 x_init 和 z_init（或直接预测 subgradient）
"""
import torch
import torch.nn as nn


class PredictionHead(nn.Module):
    """
    Prediction Head: 预测输出

    根据模式可以预测:
    - mode="subproblem": 预测 subgradient 和 opt_value（完全替代）
    - mode="warm_start": 预测 x_init 和 z_init（提供初始值）
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        mode: str = "subproblem",
    ):
        """
        Args:
            input_dim: 输入特征维度
            output_dim: 输出维度
            hidden_dim: 隐藏层维度
            mode: "subproblem" (预测 subgradient + opt_value) 或 "warm_start" (预测 x_init + z_init)
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.mode = mode

        # 共享的特征变换层
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        if mode == "subproblem":
            # 预测 subgradient 和 opt_value
            self.subgradient_head = nn.Linear(hidden_dim, output_dim)
            self.value_head = nn.Linear(hidden_dim, 1)
        elif mode == "warm_start":
            # 预测 x_init 和 z_init
            self.x_init_head = nn.Linear(hidden_dim, output_dim)
            self.z_init_head = nn.Linear(hidden_dim, output_dim)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, features: torch.Tensor) -> dict:
        """
        Args:
            features: shape (batch_size, input_dim)

        Returns:
            dict: 根据 mode 返回不同的输出
        """
        shared_features = self.shared_net(features)

        if self.mode == "subproblem":
            subgradient = self.subgradient_head(shared_features)
            opt_value = self.value_head(shared_features).squeeze(-1)
            return {
                "subgradient": subgradient,
                "opt_value": opt_value,
            }
        elif self.mode == "warm_start":
            x_init = self.x_init_head(shared_features)
            z_init = self.z_init_head(shared_features)
            return {
                "x_init": x_init,
                "z_init": z_init,
            }
