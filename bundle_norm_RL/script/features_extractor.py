"""
SB3-compatible Features Extractor for Level Bundle RL

包装 LevelBundleEncoder，输出 [sequence_embedding; global_embedding] (B, 2*hidden_dim)
支持 "deepset", "cross_attention", "self_attention" 三种编码器，通过 encoder_type 参数切换。
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from typing import Dict, Tuple

from bundle_norm_RL.script.encoder import LevelBundleEncoder


class LevelBundleFeaturesExtractor(BaseFeaturesExtractor):
    """
    Level Bundle Features Extractor

    observation_space 结构:
        - subgradient_history: (K, N_VARS+1)
        - valid_mask: (K,)
        - pi: (N_VARS,)
        - pi0: (1,)
        - lb_ub: (3,)
        - trial_point: (trial_point_dim,)
        - realization: (realization_dim,)

    输出 features_dim = 2 * hidden_dim

    Args:
        observation_space: gym Dict 空间
        hidden_dim: 编码器隐藏层维度
        encoder_type: "deepset", "cross_attention" 或 "self_attention"
        n_heads: attention 头数 (仅 attention 模式)
        n_attn_layers: attention 层数 (仅 attention 模式)
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_dim: int = 64,
        encoder_type: str = "deepset",
        n_heads: int = 4,
        n_attn_layers: int = 2,
    ):
        super().__init__(observation_space, features_dim=2 * hidden_dim)

        sg_shape = observation_space["subgradient_history"].shape
        self.K = sg_shape[0]
        self.state_dim = sg_shape[1]  # N_VARS + 1
        self.n_vars = observation_space["pi"].shape[0]
        self.trial_point_dim = observation_space["trial_point"].shape[0]
        self.realization_dim = observation_space["realization"].shape[0]
        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type

        self.encoder = LevelBundleEncoder(
            state_dim=self.state_dim,
            n_vars=self.n_vars,
            trial_point_dim=self.trial_point_dim,
            realization_dim=self.realization_dim,
            K=self.K,
            hidden_dim=hidden_dim,
            encoder_type=encoder_type,
            n_heads=n_heads,
            n_attn_layers=n_attn_layers,
        )

    def encode(self, observations: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            sequence_embedding: (B, hidden_dim)
            global_embedding: (B, hidden_dim)
        """
        return self.encoder(
            subgradient_history=observations["subgradient_history"],
            valid_mask=observations["valid_mask"],
            pi=observations["pi"],
            pi0=observations["pi0"],
            lb_ub=observations["lb_ub_norm"],
            trial_point=observations["trial_point"],
            realization=observations["realization"],
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        SB3 标准接口: 返回 [sequence_embedding; global_embedding] (B, 2*hidden_dim)
        """
        seq_emb, global_emb = self.encode(observations)
        return torch.cat([seq_emb, global_emb], dim=-1)
