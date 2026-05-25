"""
Policy Network 模块

该模块定义了自定义的 Attention-based Actor-Critic 网络组件。

===========================================
              模块职责
===========================================

1. LambdaHead: 输出每个 cut 的 lambda 权重（组件）
2. EtaHead: 输出步长 eta（组件）
3. SharedEncoderPolicy: 基于 stable-baselines3 的自定义策略（共享 encoder）
4. AttentionActorCriticPolicyNetwork: 完整的 Actor-Critic 网络（备用）

===========================================
              当前训练流程
===========================================

当前使用 stable-baselines3 的 `MultiInputActorCriticPolicy` + `AttentionFeaturesExtractor`：
- Actor 和 Critic 自动共享 `AttentionFeaturesExtractor`
- 无需使用本文件中的类（保留作为参考或备用）

如需自定义策略，可以使用 `SharedEncoderPolicy` 或 `AttentionActorCriticPolicyNetwork`。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces
from typing import Dict, Tuple

from bundle_RL.script.attention.features_extractor import AttentionFeaturesExtractor


class LambdaHead(nn.Module):
    """
    Lambda Head: 输出每个 cut 的 lambda 权重
    
    输入: cut embeddings after attention, shape (batch_size, K, hidden_dim)
    输出: lambda 分布, shape (batch_size, K)
    """
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        
        self.score_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, cut_embeddings: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
        scores = self.score_net(cut_embeddings).squeeze(-1)
        
        if valid_mask is not None:
            scores = scores.masked_fill(valid_mask == 0, float('-inf'))
        
        lambda_weights = F.softmax(scores, dim=-1)
        
        return lambda_weights


class EtaHead(nn.Module):
    """
    Eta Head: 输出步长 eta
    
    输入: global_embedding, shape (batch_size, hidden_dim)
    输出: eta, shape (batch_size, 1), 范围 (0, 1)
    """
    
    def __init__(self, hidden_dim: int = 64, scale: float = 1.0):
        super().__init__()
        
        self.scale = scale
        
        self.eta_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, global_embedding: torch.Tensor) -> torch.Tensor:
        raw_eta = self.eta_net(global_embedding)
        eta = self.scale * torch.sigmoid(raw_eta)
        return eta


class SharedEncoderPolicy(ActorCriticPolicy):
    """
    基于 stable-baselines3 的自定义策略（示例）
    
    使用共享的 AttentionFeaturesExtractor，Actor 和 Critic 共享 encoder。
    
    使用方式（在 train.py 中）:
        policy_kwargs = dict(
            features_extractor_class=AttentionFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=dict(pi=[128, 128], vf=[128, 128])
        )
        model = PPO(
            policy=SharedEncoderPolicy,  # 使用此类替代 MultiInputActorCriticPolicy
            env=env,
            policy_kwargs=policy_kwargs
        )
    
    **注意**: stable-baselines3 的 MultiInputActorCriticPolicy 已经自动实现了
    Actor 和 Critic 共享同一个 features_extractor，此类主要作为自定义策略的示例。
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
            features_extractor_kwargs = dict(features_dim=128)
        
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            **kwargs
        )


class AttentionActorCriticPolicyNetwork(nn.Module):
    """
    完整的 Actor-Critic 网络（备用方案）
    
    基于 Attention Features Extractor，输出 action 的 mean 和 log_std。
    该类作为备用方案，当前训练流程使用的是 stable-baselines3 的原生策略。
    
    如果需要完全自定义训练流程（不使用 stable-baselines3），可以使用此类。
    """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_dim: int,
        features_dim: int = 128,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        eta_scale: float = 1.0
    ):
        super().__init__()
        
        self.features_extractor = AttentionFeaturesExtractor(
            observation_space=observation_space,
            features_dim=features_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            dropout=dropout
        )
        
        self.actor = nn.Sequential(
            nn.Linear(features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        self.critic = nn.Sequential(
            nn.Linear(features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        self.actor_mean = nn.Linear(64, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        self.critic_value = nn.Linear(64, 1)
        
        self.eta_scale = eta_scale
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            observations: dict of tensors
        
        Returns:
            mean: action mean
            log_std: action log std
            value: V(s)
        """
        features = self.features_extractor(observations)
        
        actor_features = self.actor(features)
        critic_features = self.critic(features)
        
        mean = self.actor_mean(actor_features)
        log_std = self.actor_logstd.expand_as(mean)
        value = self.critic_value(critic_features)
        
        return mean, log_std, value
    
    def get_action(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        用于获取 action（训练时使用）
        
        Returns:
            action: shape (batch_size, action_dim)
        """
        mean, log_std, _ = self.forward(observations)
        std = log_std.exp()
        action = mean + std * torch.randn_like(mean)
        return action
    
    def evaluate_actions(
        self,
        observations: Dict[str, torch.Tensor],
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        用于评估 actions（训练时使用）
        
        Returns:
            log_prob: log probability of actions
            entropy: entropy of the distribution
            value: V(s)
        """
        mean, log_std, value = self.forward(observations)
        std = log_std.exp()
        
        var = std.pow(2)
        log_prob = -0.5 * ((actions - mean).pow(2) / var + 2 * log_std + 0.5 * 3.14159265359 * 2)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        entropy = 0.5 * (1 + log_std + 0.5 * 3.14159265359 * 2).sum(dim=-1)
        
        return log_prob, entropy, value