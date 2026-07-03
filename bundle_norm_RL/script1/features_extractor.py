"""
SB3-compatible Features Extractor for Incremental Level Bundle RL

包装 IncrementalLevelBundleEncoder，输出 [sequence_embedding; global_embedding] (B, 2*hidden_dim)

与 script/features_extractor.py 的区别:
    - observation_space 额外包含 pi_bar, pi0_bar
    - 使用 IncrementalLevelBundleEncoder
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from typing import Dict, Tuple

from bundle_norm_RL.script1.encoder import IncrementalLevelBundleEncoder


class IncrementalLevelBundleFeaturesExtractor(BaseFeaturesExtractor):
    """
    Incremental Level Bundle Features Extractor

    observation_space 结构:
        - subgradient_history: (K, 2*(N_VARS+1))
        - valid_mask: (K,)
        - pi_bar: (N_VARS,) - 稳定中心 pi
        - pi0_bar: (1,) - 稳定中心 pi0
        - pi: (N_VARS,)
        - pi0: (1,)
        - lb_ub_norm: (3,)
        - trial_point: (trial_point_dim,)
        - realization: (realization_dim,)

    输出 features_dim = 2 * hidden_dim

    cut_aware 模式额外属性 (在 forward 后可访问):
        - candidate_actions: (B, K, N_VARS+1) - 每个 cut 的候选增量
        - attention_weights: (B, K) - 凸组合权重
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
        self.state_dim = sg_shape[1]
        self.n_vars = observation_space["pi"].shape[0]
        self.trial_point_dim = observation_space["trial_point"].shape[0]
        self.realization_dim = observation_space["realization"].shape[0]
        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type

        self.encoder = IncrementalLevelBundleEncoder(
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

        self._candidate_actions = None
        self._attention_weights = None

    @property
    def candidate_actions(self):
        return self._candidate_actions

    @property
    def attention_weights(self):
        return self._attention_weights

    def encode(self, observations: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_emb, global_emb = self.encoder(
            subgradient_history=observations["subgradient_history"],
            valid_mask=observations["valid_mask"],
            pi_bar=observations["pi_bar"],
            pi0_bar=observations["pi0_bar"],
            pi=observations["pi"],
            pi0=observations["pi0"],
            lb_ub=observations["lb_ub_norm"],
            trial_point=observations["trial_point"],
            realization=observations["realization"],
        )

        if self.encoder_type == "cut_aware":
            self._candidate_actions = self.encoder.candidate_actions
            self._attention_weights = self.encoder.attention_weights
        else:
            self._candidate_actions = None
            self._attention_weights = None

        return seq_emb, global_emb

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        seq_emb, global_emb = self.encode(observations)
        return torch.cat([seq_emb, global_emb], dim=-1)
