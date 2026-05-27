
"""
Neural Warm Start 完整模型
整合所有模块，实现端到端的预测
"""
import torch
import torch.nn as nn
from typing import Dict, Optional

from bundle_ml.models.cut_encoder import CutEncoder
from bundle_ml.models.self_attention import SelfAttention, AttentionPooling
from bundle_ml.models.global_encoder import GlobalEncoder
from bundle_ml.models.prediction_head import PredictionHead
from bundle_ml.ml_config import MLConfig


class NeuralWarmStartModel(nn.Module):
    """
    Neural Warm Start 完整模型

    架构:
    - Cut Encoder -> Self Attention -> Attention Pooling -> Bundle Embedding
    - Global Encoder -> Global Embedding
    - (Bundle Embedding + Global Embedding) -> Shared Feature -> Prediction Head
    """

    def __init__(self, config: MLConfig = None):
        """
        Args:
            config: BundleMLConfig 对象，如果为 None 则使用默认配置
        """
        super().__init__()

        if config is None:
            config = MLConfig()

        self.config = config

        # 获取维度配置
        dims = config.dimensions
        lambda_dim = dims["lambda_dim"]
        x_prev_dim = dims["x_prev_dim"]
        realization_dim = dims["realization_dim"]
        cut_dim = dims["cut_dim"]
        output_dim = dims["output_dim"]

        # 获取网络配置
        net_config = config.network
        hidden_dim = net_config["cut_encoder_hidden_dim"]
        attention_hidden_dim = net_config["attention_hidden_dim"]
        num_heads = net_config["num_heads"]
        num_layers = net_config["num_layers"]
        ffn_dim = net_config["ffn_dim"]
        dropout = net_config["dropout"]
        prediction_head_hidden_dim = net_config["prediction_head_hidden_dim"]

        # 获取集成模式
        integration_mode = config.integration["mode"]

        # Cut Encoder
        self.cut_encoder = CutEncoder(cut_dim=cut_dim, hidden_dim=hidden_dim)

        # Self Attention Layers
        self.attention_layers = nn.ModuleList(
            [
                SelfAttention(
                    d_model=attention_hidden_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Attention Pooling
        self.attention_pooling = AttentionPooling(hidden_dim=attention_hidden_dim)

        # Global Encoder
        self.global_encoder = GlobalEncoder(
            lambda_dim=lambda_dim,
            x_prev_dim=x_prev_dim,
            realization_dim=realization_dim,
            hidden_dim=net_config["global_encoder_hidden_dim"],
        )

        # Shared Feature Dimension
        bundle_embedding_dim = net_config["bundle_embedding_dim"]
        global_embedding_dim = net_config["global_encoder_hidden_dim"]
        shared_feature_dim = bundle_embedding_dim + global_embedding_dim

        # Prediction Head
        self.prediction_head = PredictionHead(
            input_dim=shared_feature_dim,
            output_dim=output_dim,
            hidden_dim=prediction_head_hidden_dim,
            mode=integration_mode,
        )

    def forward(
        self,
        cuts: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        lambda_: torch.Tensor = None,
        x_prev: torch.Tensor = None,
        realization: torch.Tensor = None,
        stage: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            cuts: shape (batch_size, num_cuts, cut_dim)
            valid_mask: shape (batch_size, num_cuts), True 表示有效
            lambda_: shape (batch_size, lambda_dim)
            x_prev: shape (batch_size, x_prev_dim)
            realization: shape (batch_size, realization_dim)
            stage: shape (batch_size,), stage 索引

        Returns:
            dict: 预测结果
        """
        batch_size = cuts.shape[0]
        num_cuts = cuts.shape[1]
        cut_dim = cuts.shape[2]

        # ========== 添加 Dummy Token ==========
        # 创建 dummy token（可学习的参数）
        if not hasattr(self, 'dummy_token'):
            self.dummy_token = nn.Parameter(torch.randn(1, 1, cut_dim))
        
        # 确保 dummy token 在正确的设备上
        self.dummy_token = nn.Parameter(self.dummy_token.data.to(cuts.device))
        
        # 在 cuts 前面拼接 dummy token
        # cuts: (batch_size, num_cuts, cut_dim) -> (batch_size, num_cuts + 1, cut_dim)
        dummy_tokens = self.dummy_token.expand(batch_size, 1, cut_dim)
        cuts_with_dummy = torch.cat([dummy_tokens, cuts], dim=1)
        
        # 更新 valid_mask：dummy token 总是有效的
        # valid_mask: (batch_size, num_cuts) -> (batch_size, num_cuts + 1)
        if valid_mask is not None:
            dummy_valid = torch.ones(batch_size, 1, dtype=torch.bool, device=valid_mask.device)
            valid_mask_with_dummy = torch.cat([dummy_valid, valid_mask], dim=1)
        else:
            valid_mask_with_dummy = torch.ones(batch_size, num_cuts + 1, dtype=torch.bool, device=cuts.device)

        # 1. Cut Encoding
        cut_embeddings = self.cut_encoder(cuts_with_dummy)  # (batch_size, num_cuts + 1, hidden_dim)

        # 2. Self Attention
        # 创建 key_padding_mask (True 表示需要 mask)
        key_padding_mask = ~valid_mask_with_dummy

        for attention_layer in self.attention_layers:
            cut_embeddings = attention_layer(cut_embeddings, key_padding_mask=key_padding_mask)

        # 3. Attention Pooling -> Bundle Embedding
        bundle_embedding = self.attention_pooling(cut_embeddings, valid_mask=valid_mask_with_dummy)

        # 4. Global Encoding
        global_embedding = self.global_encoder(lambda_, x_prev, realization, stage)

        # 5. Concatenate Bundle and Global Embeddings
        shared_feature = torch.cat([bundle_embedding, global_embedding], dim=-1)

        # 6. Prediction Head
        predictions = self.prediction_head(shared_feature)

        return predictions

    @staticmethod
    def build_cut_features(
        cut_gradients: torch.Tensor, cut_intercepts: torch.Tensor
    ) -> torch.Tensor:
        """
        构建 cut 特征

        Args:
            cut_gradients: shape (num_cuts, gradient_dim)
            cut_intercepts: shape (num_cuts,)

        Returns:
            cut_features: shape (num_cuts, cut_dim)
        """
        # 将 intercepts 扩展为 (num_cuts, 1)
        cut_intercepts = cut_intercepts.unsqueeze(-1)
        # 拼接 g_i 和 phi_i
        cut_features = torch.cat([cut_gradients, cut_intercepts], dim=-1)
        return cut_features

    def save(self, path: str):
        """保存模型"""
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": self.config,
            },
            path,
        )

    @classmethod
    def load(cls, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        model = cls(config=checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
