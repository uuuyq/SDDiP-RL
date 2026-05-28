"""
Dirichlet 分布策略模块

该模块定义了基于 Dirichlet 分布的策略网络组件，作为原有 softmax 策略的替代方案。

===========================================
              Dirichlet 分布简介
===========================================

Dirichlet 分布是 Beta 分布的多元推广，参数为 concentration (alpha)：

    lambda ~ Dirichlet(alpha)
    E[lambda_i] = alpha_i / sum(alpha)

优势：
1. 天然满足约束：lambda_i >= 0, sum(lambda_i) = 1
2. 可建模不确定性：concentration 高 → 集中，低 → 分散
3. 与 Bundle Method 的权重组合语义契合

===========================================
              模块结构
===========================================

1. DirichletCombinedDistribution: 组合分布（lambda + eta），使用 PyTorch 原生分布
2. DirichletPolicyHead: Dirichlet 策略头
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet, Beta


class DirichletCombinedDistribution:
    """
    重构后的组合狄利克雷-贝塔分布（lambda + eta）
    完美适配 PPO 的标准数学流，支持真正的重参数化（rsample）
    """

    def __init__(
        self,
        concentration: torch.Tensor,
        eta_alpha: torch.Tensor,
        eta_beta: torch.Tensor,
        valid_mask: torch.Tensor = None
    ):
        """
        Args:
            concentration: (batch_size, K) 必须满足当 mask=0 时，concentration=1.0（使其退化为均匀分布，消除lgamma死区）
            eta_alpha / eta_beta: Beta分布的参数，用于生成连续的步长 (0, 1)
            valid_mask: (batch_size, K) 有效性掩码
        """
        self.valid_mask = valid_mask
        self._device = concentration.device

        self.lambda_dist = Dirichlet(concentration)
        self.eta_dist = Beta(eta_alpha, eta_beta)
        self.reparameterized = True

    def rsample(self):
        """
        真正的可导采样

        Returns:
            action: (batch_size, K + 1) = [lambda, eta]
        """
        lambda_sample = self.lambda_dist.rsample()

        if self.valid_mask is not None:
            lambda_sample = lambda_sample * self.valid_mask
            lambda_sample = lambda_sample / (lambda_sample.sum(dim=-1, keepdim=True) + 1e-8)

        eta_sample = self.eta_dist.rsample()
        return torch.cat([lambda_sample, eta_sample], dim=-1)

    def sample(self):
        """标准采样（不可导）"""
        return self.rsample()

    def mode(self):
        """
        推理时使用期望值作为确定性动作

        Returns:
            action: (batch_size, K + 1) = [lambda_mean, eta_mean]
        """
        lambda_mean = self.lambda_dist.mean
        if self.valid_mask is not None:
            lambda_mean = lambda_mean * self.valid_mask
            lambda_mean = lambda_mean / (lambda_mean.sum(dim=-1, keepdim=True) + 1e-8)
        return torch.cat([lambda_mean, self.eta_dist.mean], dim=-1)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """
        计算动作的对数概率

        Args:
            action: (batch_size, K + 1) = [lambda, eta]

        Returns:
            log_prob: (batch_size,)
        """
        lambda_action = action[..., :-1]
        eta_action = action[..., -1:]

        # 为规避 Dirichlet 对 0/1 边界计算 log_prob 导致的 NaN，进行微小截断
        lambda_action = lambda_action.clamp(1e-6, 1.0 - 1e-6)
        # 重新归一化，确保总和为1（Dirichlet 分布要求在 Simplex 上）
        lambda_action = lambda_action / (lambda_action.sum(dim=-1, keepdim=True) + 1e-8)

        log_prob_lambda = self.lambda_dist.log_prob(lambda_action)
        log_prob_eta = self.eta_dist.log_prob(eta_action).squeeze(-1)

        return log_prob_lambda + log_prob_eta

    def entropy(self) -> torch.Tensor:
        """混合空间下的总熵"""
        return self.lambda_dist.entropy() + self.eta_dist.entropy().squeeze(-1)


class DirichletPolicyHead(nn.Module):
    """
    Dirichlet 策略头

    输出：
    1. concentration: Dirichlet 浓度参数
    2. eta: 步长参数（使用 Beta 分布）
    """

    def __init__(
        self,
        input_dim: int,
        K: int,
        hidden_dim: int = 64
    ):
        """
        Args:
            input_dim: 输入特征维度
            K: cut 数量（lambda 维度）
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.K = K

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.alpha_head = nn.Linear(hidden_dim, K)
        self.eta_alpha_head = nn.Linear(hidden_dim, 1)
        self.eta_beta_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor = None
    ):
        """
        Args:
            x: (batch_size, input_dim) 编码器输出
            valid_mask: (batch_size, K) 有效位置掩码

        Returns:
            distribution: DirichletCombinedDistribution 对象
        """
        h = self.shared(x)

        raw_alpha = self.alpha_head(h)
        concentration = F.softplus(raw_alpha) + 1.0

        if valid_mask is not None:
            concentration = torch.where(
                valid_mask == 1,
                concentration,
                torch.ones_like(concentration)
            )

        eta_alpha = F.softplus(self.eta_alpha_head(h)) + 1.0
        eta_beta = F.softplus(self.eta_beta_head(h)) + 1.0

        return DirichletCombinedDistribution(concentration, eta_alpha, eta_beta, valid_mask)

    def get_lambda(self, x: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
        """
        获取确定性 lambda（用于推理）

        Returns:
            lambda: (batch_size, K) Dirichlet 期望值
        """
        h = self.shared(x)
        raw_alpha = self.alpha_head(h)
        concentration = F.softplus(raw_alpha) + 1.0

        if valid_mask is not None:
            concentration = torch.where(
                valid_mask == 1,
                concentration,
                torch.ones_like(concentration)
            )

        alpha_sum = concentration.sum(dim=-1, keepdim=True)
        lambdas = concentration / (alpha_sum + 1e-8)

        if valid_mask is not None:
            lambdas = lambdas * valid_mask
            lambdas_sum = lambdas.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            lambdas = lambdas / lambdas_sum

        return lambdas
