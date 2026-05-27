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

1. DirichletDistribution: 自定义 Dirichlet 分布（独立实现，不继承 torch.distributions.Distribution）
2. DirichletPolicyHead: Dirichlet 策略头
3. DirichletCombinedDistribution: 组合分布（lambda + eta）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces


class DirichletDistribution:
    """
    Dirichlet 分布实现（独立实现，不继承 torch.distributions.Distribution）

    用于 PPO 训练中的动作采样和 log_prob 计算。

    注意：由于 PPO 需要可导的采样和 log_prob，我们使用：
    - 采样：Dirichlet 期望值 + 噪声扰动（用于探索）
    - log_prob：简化的近似计算
    """

    def __init__(self, concentration: torch.Tensor, valid_mask: torch.Tensor = None):
        """
        Args:
            concentration: (batch_size, K) Dirichlet 浓度参数，必须 > 0
            valid_mask: (batch_size, K) 有效位置掩码，1 表示有效
        """
        self.concentration = concentration
        self.valid_mask = valid_mask
        self._device = concentration.device

        # 计算 Dirichlet 期望值
        alpha_sum = concentration.sum(dim=-1, keepdim=True)
        self._mean = concentration / (alpha_sum + 1e-8)

        # 应用 valid_mask：无效位置设为 0
        if valid_mask is not None:
            self._mean = self._mean * valid_mask
            # 重新归一化（确保有效位置和为1）
            mean_sum = self._mean.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            self._mean = self._mean / mean_sum

        self.batch_shape = concentration.shape[:-1]
        self.event_shape = concentration.shape[-1:]

    @property
    def mean(self):
        """返回 Dirichlet 期望值"""
        return self._mean

    def rsample(self, sample_shape=torch.Size()):
        """
        重参数化采样（可导）

        使用期望值 + Dirichlet 噪声实现可导采样
        """
        # 基础：使用 Dirichlet 期望值
        base = self._mean

        # 添加 Dirichlet 噪声实现探索
        # 噪声强度与 concentration 成反比（浓度高时噪声小）
        concentration_sum = self.concentration.sum(dim=-1, keepdim=True)
        noise_scale = 0.1 / (concentration_sum.sqrt() + 1e-8)

        # 从 Dirichlet 采样噪声（使用 Gamma 分布）
        noise = torch._standard_gamma(self.concentration)
        noise = noise / (noise.sum(dim=-1, keepdim=True) + 1e-8)

        # 组合：主要用期望值，少量加噪声
        sampled = base + noise_scale * (noise - 1.0 / self.concentration.shape[-1])

        # 应用 valid_mask 并重新归一化
        if self.valid_mask is not None:
            sampled = sampled * self.valid_mask
            sampled_sum = sampled.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            sampled = sampled / sampled_sum

        return sampled.clamp(min=1e-8, max=1 - 1e-8)

    def sample(self, sample_shape=torch.Size()):
        """标准采样（不可导）"""
        return self.rsample(sample_shape)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        计算 log probability

        Args:
            value: (batch_size, K) 动作值

        Returns:
            log_prob: (batch_size,) 对数概率
        """
        eps = 1e-8
        value = value.clamp(min=eps, max=1 - eps)

        # Dirichlet log probability
        log_x = torch.log(value)
        log_prob = ((self.concentration - 1) * log_x).sum(dim=-1)

        # 添加 Dirichlet 归一化项
        alpha_sum = self.concentration.sum(dim=-1)
        log_normalizer = torch.lgamma(self.concentration).sum(dim=-1) - torch.lgamma(alpha_sum)
        log_prob = log_prob - log_normalizer

        # 处理无效位置：只对有效位置计算 log_prob
        if self.valid_mask is not None:
            # 无效位置的贡献置零
            valid_value = value * self.valid_mask + eps * (1 - self.valid_mask)
            valid_log_x = torch.log(valid_value)
            valid_log_prob = ((self.concentration - 1) * valid_log_x).sum(dim=-1)
            log_prob = valid_log_prob - log_normalizer

        return log_prob

    def entropy(self) -> torch.Tensor:
        """
        计算熵

        Dirichlet 熵：H = log B(alpha) + (alpha_0 - K) * psi(alpha_0)
                      - sum((alpha_i - 1) * psi(alpha_i))
        其中 alpha_0 = sum(alpha), psi 是 digamma 函数
        """
        K = self.concentration.shape[-1]
        alpha_0 = self.concentration.sum(dim=-1, keepdim=True)

        # Digamma 函数近似
        def digamma(x):
            return torch.log(x + 1e-8) - 1 / (2 * x + 1e-8)

        log_B_alpha = torch.lgamma(self.concentration).sum(dim=-1) - torch.lgamma(alpha_0.squeeze(-1))
        term1 = log_B_alpha
        term2 = (alpha_0 - K) * digamma(alpha_0.squeeze(-1))
        term3 = ((self.concentration - 1) * digamma(self.concentration)).sum(dim=-1)

        entropy = term1 + term2 - term3

        return entropy


class DirichletPolicyHead(nn.Module):
    """
    Dirichlet 策略头

    输出：
    1. concentration: Dirichlet 浓度参数
    2. eta: 步长参数
    """

    def __init__(
        self,
        input_dim: int,
        K: int,
        hidden_dim: int = 64,
        min_alpha: float = 1.0,
        eta_scale: float = 1.0
    ):
        """
        Args:
            input_dim: 输入特征维度（通常为 hidden_dim 或 features_dim）
            K: cut 数量（lambda 维度）
            hidden_dim: 隐藏层维度
            min_alpha: 最小 concentration 值，确保分布不会过于集中
            eta_scale: 步长缩放因子
        """
        super().__init__()
        self.K = K
        self.min_alpha = min_alpha
        self.eta_scale = eta_scale

        # 共享特征层
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Concentration 参数输出（使用 softplus 确保 > 0）
        self.alpha_head = nn.Linear(hidden_dim, K)

        # Eta（步长）输出
        self.eta_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor = None
    ) -> tuple:
        """
        Args:
            x: (batch_size, input_dim) 编码器输出
            valid_mask: (batch_size, K) 有效位置掩码

        Returns:
            concentration: (batch_size, K) Dirichlet 浓度参数
            eta: (batch_size, 1) 步长，范围 (0, eta_scale)
            distribution: DirichletDistribution 对象
        """
        h = self.shared(x)

        # Concentration 参数（使用 softplus 确保 > 0）
        raw_alpha = self.alpha_head(h)
        concentration = F.softplus(raw_alpha) + self.min_alpha

        # 应用 valid_mask
        if valid_mask is not None:
            concentration = concentration * valid_mask + 1e-6 * (1 - valid_mask)

        # Eta 步长
        raw_eta = self.eta_head(h)
        eta = self.eta_scale * torch.sigmoid(raw_eta)

        # 创建分布对象
        distribution = DirichletDistribution(concentration, valid_mask)

        return concentration, eta, distribution

    def get_lambda(self, x: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
        """
        获取确定性 lambda（用于推理）

        Returns:
            lambda: (batch_size, K) Dirichlet 期望值
        """
        concentration, _, _ = self.forward(x, valid_mask)
        alpha_sum = concentration.sum(dim=-1, keepdim=True)
        lambdas = concentration / (alpha_sum + 1e-8)

        if valid_mask is not None:
            lambdas = lambdas * valid_mask
            lambdas_sum = lambdas.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            lambdas = lambdas / lambdas_sum

        return lambdas


class EtaDistribution:
    """
    Eta（步长）的分布实现

    使用简化版本：固定值 + 小噪声
    """

    def __init__(self, eta: torch.Tensor):
        """
        Args:
            eta: (batch_size, 1) 步长值，范围 (0, 1)
        """
        self.eta = eta
        self._device = eta.device
        self.batch_shape = eta.shape[:-1]
        self.event_shape = eta.shape[-1:]

    def rsample(self, sample_shape=torch.Size()):
        """重参数化采样"""
        # 添加轻微噪声
        noise = torch.randn_like(self.eta) * 0.1
        eta_noisy = self.eta + noise
        return torch.sigmoid(eta_noisy).clamp(0.01, 0.99)

    def sample(self, sample_shape=torch.Size()):
        """标准采样"""
        return self.eta

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """log probability（简化为 uniform prior）"""
        return torch.zeros_like(value.squeeze(-1))

    def entropy(self) -> torch.Tensor:
        """熵（简化为常数）"""
        return torch.zeros_like(self.eta.squeeze(-1))


class DirichletCombinedDistribution:
    """
    组合 Dirichlet 分布（lambda + eta）

    用于 PPO 的 action 采样和 log_prob 计算
    """

    def __init__(
        self,
        concentration: torch.Tensor,
        eta: torch.Tensor,
        valid_mask: torch.Tensor = None,
        lambda_temp: float = 1.0
    ):
        """
        Args:
            concentration: (batch_size, K) Dirichlet 浓度参数
            eta: (batch_size, 1) 步长
            valid_mask: (batch_size, K) 有效位置掩码
            lambda_temp: 温度参数，控制采样多样性
        """
        self.lambda_dist = DirichletDistribution(concentration, valid_mask)
        self.eta_dist = EtaDistribution(eta)
        self.valid_mask = valid_mask
        self.lambda_temp = lambda_temp
        # SB3 需要的属性
        self.reparameterized = True
        self._device = concentration.device

    def sample(self):
        """
        采样动作（不可导）

        Returns:
            action: (batch_size, K + 1) = [lambda, eta]
        """
        return self.rsample()

    def rsample(self):
        """
        重参数化采样（可导）

        Returns:
            action: (batch_size, K + 1) = [lambda, eta]
        """
        lambda_sample = self.lambda_dist.rsample()
        eta_sample = self.eta_dist.rsample()
        return torch.cat([lambda_sample, eta_sample], dim=-1)

    def mode(self):
        """
        返回最可能的动作（确定性）

        Returns:
            action: (batch_size, K + 1) = [lambda_mean, eta_mean]
        """
        lambda_mode = self.lambda_dist.mean
        eta_mode = self.eta_dist.eta
        return torch.cat([lambda_mode, eta_mode], dim=-1)

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

        log_prob_lambda = self.lambda_dist.log_prob(lambda_action)
        log_prob_eta = self.eta_dist.log_prob(eta_action)

        return log_prob_lambda + log_prob_eta.squeeze(-1)

    def entropy(self) -> torch.Tensor:
        """计算熵"""
        return self.lambda_dist.entropy() + self.eta_dist.entropy().squeeze(-1)
