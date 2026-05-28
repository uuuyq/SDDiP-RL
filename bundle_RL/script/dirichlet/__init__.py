"""
Dirichlet 分布策略模块

基于 Dirichlet 分布的强化学习策略实现，用于 Bundle Method 的权重组合优化。

模块结构：
- dirichlet_policy: Dirichlet 分布策略核心实现
- train_dirichlet: 训练逻辑和模型定义
- env: 环境封装
- encoder_v2: Attention 编码器（带 mask 支持）
- features_extractor: 特征提取器
"""

from bundle_RL.script.dirichlet.dirichlet_policy import (
    DirichletPolicyHead,
    DirichletCombinedDistribution
)

from bundle_RL.script.dirichlet.train_dirichlet import (
    DirichletFeaturesExtractor,
    DirichletExtractorWithCuts,
    train
)

from bundle_RL.script.dirichlet.env import BundleDualEnv

__all__ = [
    # 分布类
    'DirichletPolicyHead',
    'DirichletCombinedDistribution',
    
    # 特征提取器
    'DirichletFeaturesExtractor',
    'DirichletExtractorWithCuts',
    
    # 环境
    'BundleDualEnv',
    
    # 训练函数
    'train',
]
