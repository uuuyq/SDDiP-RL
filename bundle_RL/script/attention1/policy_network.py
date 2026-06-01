"""
Policy Network 模块（简化版）

该模块定义了 attention1 简化版的自定义 Actor-Critic 策略组件。

===========================================
              模块职责
===========================================

1. LambdaHead: 利用 Q-K 注意力对 cut_embeddings 打分（scores 即 lambda_mean）
2. EtaHead:    用 [global; CLS] 输出 eta_mean
3. ValueHead:  用 [global; CLS] 输出 V(s)
4. AttentionActorCriticPolicy:
   - 继承 stable_baselines3.common.policies.ActorCriticPolicy
   - 复用父类的 action_dist (DiagGaussianDistribution) 与 log_std (nn.Parameter)
   - 重写 forward / evaluate_actions / predict_values / _predict
   - 在 forward 中绕过 mlp_extractor / action_net / value_net，
     直接调用 features_extractor.encode(obs) 得到 (H, global, CLS)，
     再走 LambdaHead / EtaHead / ValueHead 自定义路径

===========================================
              动作分布
===========================================

    action_mean = [LambdaHead(h, H), EtaHead(h)]   shape (B, K+1)
    log_std     = self.log_std                     shape (K+1,) (父类创建)
    dist        = DiagGaussianDistribution.proba_distribution(action_mean, log_std)
    action      = dist.sample() / dist.mode()
    log_prob    = dist.log_prob(action)            shape (B,)
    entropy     = dist.entropy()                   shape (B,)

===========================================
              环境侧约定
===========================================

    Policy 输出的 action 是 raw_lambda 与 raw_eta 的拼接（未做 mask/softmax/sigmoid）。
    BundleDualEnv.step 内部完成：
        raw_lambda → mask_fill(-inf) → softmax → lambdas
        raw_eta    → sigmoid → eta
"""
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces
from typing import Dict, Tuple

from bundle_RL.script.attention1.features_extractor import AttentionFeaturesExtractor


# ============================
# LambdaHead
# ============================

class LambdaHead(nn.Module):
    """
    Lambda Head: 利用 Q-K 注意力对 cut_embeddings 打分

    输入:
        h_combined: (B, 2H) - [global ; CLS]
        cut_embeddings: (B, K, H) - 即 H

    输出:
        lambda_mean: (B, K) - raw scores（未 mask、未 softmax）

    流程:
        q = QueryNet(h_combined)               (B, H)
        scores = q @ Hᵀ / sqrt(H)              (B, K)
        scores = tanh(scores) * 3.0            (B, K)  # 软约束在 (-3, 3)，防数值漂移
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.scale = hidden_dim ** 0.5

        self.query_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(
        self,
        h_combined: torch.Tensor,
        cut_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h_combined: (B, 2H)
            cut_embeddings: (B, K, H)

        Returns:
            scores: (B, K)
        """
        q = self.query_net(h_combined)              # (B, H)
        q = q.unsqueeze(1)                          # (B, 1, H)
        scores = torch.bmm(q, cut_embeddings.transpose(1, 2))  # (B, 1, K)
        scores = scores.squeeze(1) / self.scale     # (B, K)
        # 不再使用 tanh 压缩。
        # action_space 已扩展到 [-10, 10]，配合 target_kl + max_grad_norm
        # 足以约束增长速度，同时保留 softmax 产生集中分布的能力。
        return scores


# ============================
# EtaHead
# ============================

class EtaHead(nn.Module):
    """
    Eta Head: 输出步长 eta 的均值

    输入:
        h_combined: (B, 2H) - [global ; CLS]

    输出:
        eta_mean: (B, 1) - raw 均值（sigmoid 在 env 中完成）
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h_combined: torch.Tensor) -> torch.Tensor:
        return self.net(h_combined)


# ============================
# ValueHead
# ============================

class ValueHead(nn.Module):
    """
    Value Head: 输出状态值 V(s)

    输入:
        h_combined: (B, 2H) - [global ; CLS]

    输出:
        value: (B, 1)
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h_combined: torch.Tensor) -> torch.Tensor:
        return self.net(h_combined)


# ============================
# AttentionActorCriticPolicy
# ============================

class AttentionActorCriticPolicy(ActorCriticPolicy):
    """
    自定义 Actor-Critic 策略（简化版）

    继承自 stable_baselines3.common.policies.ActorCriticPolicy：
    - 复用父类的 action_dist (DiagGaussianDistribution) 与 log_std (nn.Parameter)
    - 重写 forward / evaluate_actions / predict_values / _predict，
      直接调用 features_extractor.encode(obs) 得到 (H, global, CLS)，
      再走 LambdaHead / EtaHead / ValueHead 自定义路径

    使用方式（在 train.py 中）:
        policy_kwargs = dict(
            features_extractor_class=AttentionFeaturesExtractor,
            features_extractor_kwargs=dict(
                hidden_dim=64, num_heads=4, num_layers=1, ffn_dim=128, dropout=0.1
            ),
            net_arch=[],   # 不需要 mlp_extractor 中间层
        )
        model = PPO(
            policy=AttentionActorCriticPolicy,
            env=env,
            policy_kwargs=policy_kwargs
        )

    备注:
        父类会创建 mlp_extractor / action_net / value_net 等模块，
        本 Policy 在 forward 中不调用它们；这些模块对应的参数会被注册到 optimizer，
        但梯度始终为零（因为没有路径连到 loss），不影响训练正确性，
        仅产生少量额外内存占用。
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        lr_schedule,
        features_extractor_class=AttentionFeaturesExtractor,
        features_extractor_kwargs=None,
        **kwargs
    ):
        if features_extractor_kwargs is None:
            features_extractor_kwargs = dict(hidden_dim=64)

        # 强制 net_arch=[]，避免 mlp_extractor 中间层（不会被使用）
        if "net_arch" not in kwargs:
            kwargs["net_arch"] = []

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            **kwargs
        )

        # 从 features_extractor 中读取 hidden_dim 用于构造 head
        hidden_dim = self.features_extractor.hidden_dim

        # 自定义 head
        self.lambda_head = LambdaHead(hidden_dim)
        self.eta_head = EtaHead(hidden_dim)
        self.value_head = ValueHead(hidden_dim)

        # 将自定义 head 注册到优化器（父类已用 self.parameters() 创建 optimizer，
        # 这里手动重建以包含新加入的 head）
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs
        )

    # --------------------------------------------------------------
    # 编码 + head 计算
    # --------------------------------------------------------------

    def _encode(
        self,
        obs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        调用 features_extractor.encode 获取三元组，并构造 h = [global ; CLS]

        Returns:
            cut_embeddings: (B, K, H)
            h_combined:    (B, 2H)
            global_embedding: (B, H)
            cls_embedding:    (B, H)
        """
        cut_embeddings, global_embedding, cls_embedding = self.features_extractor.encode(obs)
        h_combined = torch.cat([global_embedding, cls_embedding], dim=-1)
        return cut_embeddings, h_combined, global_embedding, cls_embedding

    def _compute_action_mean(
        self,
        h_combined: torch.Tensor,
        cut_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        action_mean = [lambda_mean ; eta_mean]，shape (B, K+1)
        """
        lambda_mean = self.lambda_head(h_combined, cut_embeddings)   # (B, K)
        eta_mean = self.eta_head(h_combined)                         # (B, 1)
        return torch.cat([lambda_mean, eta_mean], dim=-1)

    # --------------------------------------------------------------
    # 重写 SB3 策略接口
    # --------------------------------------------------------------

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        重写 forward：构造动作分布、采样动作、计算 V(s)

        Returns:
            actions: (B, K+1)
            values:  (B, 1)
            log_prob: (B,)
        """
        cut_embeddings, h_combined, _, _ = self._encode(obs)

        action_mean = self._compute_action_mean(h_combined, cut_embeddings)
        distribution = self.action_dist.proba_distribution(action_mean, self.log_std)

        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        values = self.value_head(h_combined)

        return actions, values, log_prob

    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        重写 evaluate_actions：用于 PPO 的 ratio 计算

        Returns:
            values: (B, 1)
            log_prob: (B,)
            entropy: (B,)
        """
        cut_embeddings, h_combined, _, _ = self._encode(obs)

        action_mean = self._compute_action_mean(h_combined, cut_embeddings)
        distribution = self.action_dist.proba_distribution(action_mean, self.log_std)

        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        values = self.value_head(h_combined)

        return values, log_prob, entropy

    def predict_values(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        重写 predict_values：用于 GAE / value bootstrap
        """
        _, h_combined, _, _ = self._encode(obs)
        return self.value_head(h_combined)

    def _predict(
        self,
        observation: Dict[str, torch.Tensor],
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        重写 _predict：用于 model.predict（不需要返回 value/log_prob）
        """
        cut_embeddings, h_combined, _, _ = self._encode(observation)
        action_mean = self._compute_action_mean(h_combined, cut_embeddings)
        distribution = self.action_dist.proba_distribution(action_mean, self.log_std)
        return distribution.get_actions(deterministic=deterministic)

    def get_distribution(self, obs: Dict[str, torch.Tensor]):
        """
        重写 get_distribution：返回当前观察下的动作分布
        """
        cut_embeddings, h_combined, _, _ = self._encode(obs)
        action_mean = self._compute_action_mean(h_combined, cut_embeddings)
        return self.action_dist.proba_distribution(action_mean, self.log_std)
