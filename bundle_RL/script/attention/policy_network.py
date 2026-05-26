"""
Policy Network 模块

该模块定义了自定义的 Attention-based Actor-Critic 网络组件。

===========================================
              模块职责
===========================================

1. LambdaHead: 输出每个 cut 的 lambda 权重（组件）
2. EtaHead: 输出步长 eta（组件）
3. SharedEncoderPolicy: 基于 stable-baselines3 的自定义策略（共享 encoder）
4. SeparateEncoderPolicy: 基于 stable-baselines3 的自定义策略（分离 encoder）
5. AttentionActorCriticPolicyNetwork: 完整的 Actor-Critic 网络（备用）

===========================================
              当前训练流程
===========================================

当前使用 stable-baselines3 的 `MultiInputActorCriticPolicy` + `AttentionFeaturesExtractor`：
- Actor 和 Critic 自动共享 `AttentionFeaturesExtractor`
- 如需不共享 encoder，可使用 `SeparateEncoderPolicy`

===========================================
              策略选择指南
===========================================

| 策略类型 | 适用场景 | 优势 |
|----------|----------|------|
| SharedEncoderPolicy | 默认推荐 | 参数共享，训练稳定，数据效率高 |
| SeparateEncoderPolicy | 需要独立优化 | Actor/Critic 可独立学习不同特征 |
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
    基于 stable-baselines3 的自定义策略（共享 encoder）
    
    Actor 和 Critic 共享同一个 AttentionFeaturesExtractor。
    
    使用方式（在 train.py 中）:
        policy_kwargs = dict(
            features_extractor_class=AttentionFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=dict(pi=[128, 128], vf=[128, 128])
        )
        model = PPO(
            policy=SharedEncoderPolicy,
            env=env,
            policy_kwargs=policy_kwargs
        )
    
    **注意**: stable-baselines3 的 MultiInputActorCriticPolicy 已经自动实现了
    Actor 和 Critic 共享同一个 features_extractor，此类主要作为参考。
    """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        lr_schedule,
        features_extractor_class=AttentionFeaturesExtractor,
        features_extractor_kwargs=None,** kwargs
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


class SeparateEncoderPolicy(ActorCriticPolicy):
    """
    基于 stable-baselines3 的自定义策略（分离 encoder）
    
    Actor 和 Critic 使用独立的 AttentionFeaturesExtractor，不共享参数。
    
    使用方式（在 train.py 中）:
        policy_kwargs = dict(
            features_extractor_class=AttentionFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=dict(pi=[128, 128], vf=[128, 128])
        )
        model = PPO(
            policy=SeparateEncoderPolicy,  # 使用此类替代 MultiInputActorCriticPolicy
            env=env,
            policy_kwargs=policy_kwargs
        )
    
    **适用场景**: 当需要 Actor 和 Critic 学习不同的特征表示时使用。
    注意：参数数量翻倍，训练数据需求更大。
    """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        lr_schedule,
        features_extractor_class=AttentionFeaturesExtractor,
        features_extractor_kwargs=None,** kwargs
    ):
        if features_extractor_kwargs is None:
            features_extractor_kwargs = dict(features_dim=128)
        
        # 先调用父类初始化（会创建一个共享的 features_extractor，用于 Actor）
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,** kwargs
        )
        
        # 创建独立的 Critic features_extractor
        self.critic_features_extractor = features_extractor_class(
            observation_space=observation_space,
            **features_extractor_kwargs
        )
        
        # 将 critic_features_extractor 注册为模块
        self.add_module("critic_features_extractor", self.critic_features_extractor)
    
    def extract_features(self, obs, features_extractor=None):
        """
        提取特征（兼容父类接口）
        
        Args:
            obs: 观察（dict 或 tensor）
            features_extractor: 可选，指定使用哪个 features_extractor
                               默认为 None，使用 Actor 的 features_extractor
        """
        if features_extractor is None:
            features_extractor = self.features_extractor
        
        if isinstance(obs, dict):
            return features_extractor(obs)
        return features_extractor(obs.unsqueeze(0)).squeeze(0)
    
    def forward(self, obs, deterministic=False):
        """
        重写 forward 方法，使用分离的 encoder
        
        Returns:
            actions: 动作
            values: 价值估计
            log_probs: 动作的对数概率
        """
        # Actor 使用共享的 features_extractor（由父类创建）
        actor_features = self.extract_features(obs, self.features_extractor)
        
        # Critic 使用独立的 features_extractor
        critic_features = self.extract_features(obs, self.critic_features_extractor)
        
        # Actor 分支：使用父类的 get_distribution 保证一致性
        latent_pi = self.mlp_extractor.forward_actor(actor_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.sample()
        
        # 计算 log_probs
        log_probs = distribution.log_prob(actions)
        
        # Critic 分支
        latent_vf = self.mlp_extractor.forward_critic(critic_features)
        values = self.value_net(latent_vf)
        
        return actions, values, log_probs
    
    def evaluate_actions(self, obs, actions):
        """
        重写 evaluate_actions 方法，使用分离的 encoder
        
        Returns:
            values: 价值估计
            log_prob: 动作的对数概率
            entropy: 熵
        """
        # Actor 使用共享的 features_extractor
        actor_features = self.extract_features(obs, self.features_extractor)
        
        # Critic 使用独立的 features_extractor
        critic_features = self.extract_features(obs, self.critic_features_extractor)
        
        # Actor 分支：使用父类的方法保证与 forward 一致
        latent_pi = self.mlp_extractor.forward_actor(actor_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        # 计算 log_prob 和 entropy
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        # Critic 分支
        latent_vf = self.mlp_extractor.forward_critic(critic_features)
        values = self.value_net(latent_vf)
        
        return values, log_prob, entropy
    
    def predict_values(self, obs):
        """
        重写 predict_values 方法，使用独立的 Critic features_extractor
        
        Args:
            obs: 观察（dict）
        
        Returns:
            values: 价值估计
        """
        critic_features = self.extract_features(obs, self.critic_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(critic_features)
        return self.value_net(latent_vf)


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