"""
Policy Network 模块

该模块定义了自定义的 DeepSet-based Actor-Critic 网络组件。

===========================================
              模块职责
===========================================

1. QueryNetwork: 生成查询向量 q（输入 z_g）
2. KeyNetwork: 生成每个 cut 的 key 向量 k（输入 h）
3. EtaHead: 输出步长 eta（输入 z_g）
4. DeepSetActorCriticPolicy: 基于 stable-baselines3 的自定义策略

===========================================
              当前训练流程
===========================================

┌─────────────────────────────────────────────────────────────────┐
│                    DeepSetActorCriticPolicy                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           DeepSetFeaturesExtractor (共享)               │   │
│  │                     ↓                                 │   │
│  │           h, z_g = extractor(obs)                     │   │
│  └───────────────────────────┬───────────────────────────┘   │
│                              ↓                               │
│         ┌────────────────────┴────────────────────┐          │
│         ↓                                         ↓          │
│  ┌───────────────────┐                  ┌──────────────┐     │
│  │       Actor       │                  │   Critic     │     │
│  │  ┌─────────────┐  │                  │              │     │
│  │  │QueryNetwork │  │                  │              │     │
│  │  │KeyNetwork   │→│ cut_weights       │              │     │
│  │  │EtaHead      │→│ eta               │              │     │
│  │  └─────────────┘  │                  └──────────────┘     │
│  └───────────────────┘                  ↓                     │
│         ↓                              V(z_g)                 │
│  ┌──────────────┐                                             │
│  │  action_net  │                                             │
│  │ [cut_weights, eta]                                         │
│  └──────────────┘                                             │
└─────────────────────────────────────────────────────────────────┘

===========================================
              策略选择指南
===========================================

| 策略类型 | 适用场景 | 优势 |
|----------|----------|------|
| DeepSetActorCriticPolicy | 默认推荐 | 符合 DeepSet 架构，参数共享 |
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces
from typing import Dict, Tuple

from bundle_RL.script.deepset.features_extractor import DeepSetFeaturesExtractor


class QueryNetwork(nn.Module):
    """
    Query Network: 生成查询向量

    输入: z_g = [h_g, pi_emb, tp_emb, rlz_emb]
    输出: q (query_dim,)

    含义: "当前优化状态需要什么方向？"
    """

    def __init__(self, input_dim: int, query_dim: int = 64):
        super().__init__()

        self.query_net = nn.Sequential(
            nn.Linear(input_dim, query_dim),
            nn.ReLU(),
            nn.Linear(query_dim, query_dim),
            nn.Sigmoid()  # 输出限制在 (0, 1)
        )

    def forward(self, z_g: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_g: shape (batch_size, input_dim)

        Returns:
            q: shape (batch_size, query_dim)
        """
        return self.query_net(z_g)


class KeyNetwork(nn.Module):
    """
    Key Network: 生成每个 cut 的 key 向量

    输入: h_i (hidden_dim,)
    输出: k_i (key_dim,)

    含义: "这个 cut 能提供什么信息？"
    """

    def __init__(self, hidden_dim: int, key_dim: int = 64):
        super().__init__()

        self.key_net = nn.Sequential(
            nn.Linear(hidden_dim, key_dim),
            nn.ReLU(),
            nn.Linear(key_dim, key_dim),
            nn.Sigmoid()  # 输出限制在 (0, 1)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: shape (batch_size, K, hidden_dim)

        Returns:
            keys: shape (batch_size, K, key_dim)
        """
        return self.key_net(h)


class EtaHead(nn.Module):
    """
    Eta Head: 输出步长 eta 的原始值

    输入: z_g (全局特征), shape (batch_size, z_g_dim)
    输出: raw_eta, shape (batch_size, 1)
    
    注意：不应用 sigmoid，由环境的 step 方法处理 sigmoid 映射
    """
    
    def __init__(self, input_dim: int):
        super().__init__()
        
        self.eta_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1)
        )
    
    def forward(self, z_g: torch.Tensor) -> torch.Tensor:
        raw_eta = self.eta_net(z_g)
        return raw_eta


class DeepSetActorCriticPolicy(ActorCriticPolicy):
    """
    DeepSet-based Actor-Critic Policy

    核心特点:
    - FeatureExtractor 输出 h (各cut编码) 和 z_g (全局特征)
    - Actor 使用 QueryNetwork(z_g) 和 KeyNetwork(h) 计算 cut_weights
    - Actor 使用 EtaHead(z_g) 输出步长 eta
    - Critic 使用 z_g 评估状态价值 V(s)
    """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        lr_schedule,
        features_extractor_class=DeepSetFeaturesExtractor,
        features_extractor_kwargs=None,
        query_dim: int = 64,
        key_dim: int = 64,** kwargs
    ):
        if features_extractor_kwargs is None:
            features_extractor_kwargs = dict(features_dim=128)
        
        # 获取 K 值（用于 action 维度）
        K = observation_space["cuts"].shape[0]
        self._K = K
        
        # 保存参数供后续使用
        self._query_dim = query_dim
        self._key_dim = key_dim
        
        # 获取 hidden_dim 和计算 z_g_dim
        hidden_dim = features_extractor_kwargs.get("hidden_dim", 64)
        z_g_dim = 4 * hidden_dim
        
        # 修改 features_extractor_kwargs，将 features_dim 设置为 z_g_dim
        # 这样 mlp_extractor 会使用正确的输入维度
        features_extractor_kwargs = features_extractor_kwargs.copy()
        features_extractor_kwargs['features_dim'] = z_g_dim
        
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,** kwargs
        )
        
        # Actor 组件
        self.query_net = QueryNetwork(input_dim=z_g_dim, query_dim=query_dim)
        self.key_net = KeyNetwork(hidden_dim=hidden_dim, key_dim=key_dim)
        self.eta_head = EtaHead(input_dim=z_g_dim)  # 不应用 sigmoid，由环境处理

        # 可学习的 log_std 参数（用于动作分布）
        action_dim = K + 1  # cut_weights (K) + eta (1)
        # 初始化为 -2.0，使初始 std = exp(-2.0) ≈ 0.14，降低动作方差
        self.actor_logstd = nn.Parameter(torch.full((1, action_dim), -2.0))

        # 注册为模块
        self.add_module("query_net", self.query_net)
        self.add_module("key_net", self.key_net)
        self.add_module("eta_head", self.eta_head)

    def _get_action_dist_from_latent(self, latent_pi, latent_sde=None):
        """
        重写父类方法，直接使用 action mean 创建动作分布

        Args:
            latent_pi: 已经计算好的 action mean (batch_size, action_dim)
            latent_sde: 用于 SDE 策略（这里不需要）

        Returns:
            动作分布对象
        """
        action_mean = latent_pi
        log_std = self.actor_logstd.expand_as(action_mean)

        distribution = self.action_dist.proba_distribution(action_mean, log_std)

        return distribution
    
    def forward(self, obs, deterministic=False):
        """
        重写 forward 方法
        """
        # 提取特征（会保存 h 和 z_g）
        z_g = self.extract_features(obs)
        h = self.features_extractor.h
        
        # Actor 分支：计算 cut_weights 和 eta
        q = self.query_net(z_g)  # (batch_size, query_dim)
        k = self.key_net(h)      # (batch_size, K, key_dim)
        
        # 计算权重 score_i = q^T k_i
        scores = torch.bmm(q.unsqueeze(1), k.transpose(1, 2)).squeeze(1)  # (batch_size, K)
        # 应用 sigmoid 将 cut_weights 限制在 [0, 1]
        cut_weights = torch.sigmoid(scores)  # (batch_size, K)
        
        # 计算步长 eta（应用 sigmoid 限制在 [0, 1]）
        eta = torch.sigmoid(self.eta_head(z_g))  # (batch_size, 1)
        
        # 拼接 action（输出已归一化到 [0, 1]）
        action_mean = torch.cat([cut_weights, eta], dim=-1)  # (batch_size, K+1)
        
        # 获取动作分布
        distribution = self._get_action_dist_from_latent(action_mean)
        
        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.sample()
        
        log_probs = distribution.log_prob(actions)
        
        # Critic 分支：使用 z_g 评估价值
        latent_vf = self.mlp_extractor.forward_critic(z_g)
        values = self.value_net(latent_vf)
        
        return actions, values, log_probs
    
    def evaluate_actions(self, obs, actions):
        """
        重写 evaluate_actions 方法
        """
        # 提取特征
        z_g = self.extract_features(obs)
        h = self.features_extractor.h
        
        # Actor 分支：计算 action mean
        q = self.query_net(z_g)
        k = self.key_net(h)
        
        scores = torch.bmm(q.unsqueeze(1), k.transpose(1, 2)).squeeze(1)
        # 应用 sigmoid 将 cut_weights 限制在 [0, 1]
        cut_weights = torch.sigmoid(scores)
        
        # 计算步长 eta（应用 sigmoid 限制在 [0, 1]）
        eta = torch.sigmoid(self.eta_head(z_g))
        action_mean = torch.cat([cut_weights, eta], dim=-1)
        
        # 获取分布
        distribution = self._get_action_dist_from_latent(action_mean)
        
        # 计算 log_prob 和 entropy
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        # Critic 分支
        latent_vf = self.mlp_extractor.forward_critic(z_g)
        values = self.value_net(latent_vf)
        
        return values, log_prob, entropy
    
    def predict_values(self, obs):
        """
        重写 predict_values 方法
        """
        z_g = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(z_g)
        return self.value_net(latent_vf)