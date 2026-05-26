
"""
测试 Neural Warm Start 模型
"""
import numpy as np
import torch

from bundle_ml import NeuralWarmStartModel, MLConfig


def test_model_forward():
    """测试模型前向传播"""
    print("=" * 60)
    print("Testing Neural Warm Start Model Forward Pass")
    print("=" * 60)

    # 加载配置
    config = MLConfig()
    dims = config.dimensions

    # 创建模型
    model = NeuralWarmStartModel(config=config)
    model.eval()

    # 准备测试数据
    batch_size = 2
    num_cuts = 10
    lambda_dim = dims["lambda_dim"]
    x_prev_dim = dims["x_prev_dim"]
    realization_dim = dims["realization_dim"]
    cut_dim = dims["cut_dim"]

    # 随机生成输入
    cuts = torch.randn(batch_size, num_cuts, cut_dim)
    valid_mask = torch.ones(batch_size, num_cuts, dtype=torch.bool)
    valid_mask[:, -3:] = False  # 最后 3 个 cuts 无效

    lambda_ = torch.randn(batch_size, lambda_dim)
    x_prev = torch.randn(batch_size, x_prev_dim)
    realization = torch.randn(batch_size, realization_dim)
    stage = torch.randint(0, 10, (batch_size,))

    # 前向传播
    with torch.no_grad():
        predictions = model(
            cuts=cuts,
            valid_mask=valid_mask,
            lambda_=lambda_,
            x_prev=x_prev,
            realization=realization,
            stage=stage,
        )

    # 检查输出
    print(f"\nInput shapes:")
    print(f"  cuts: {cuts.shape}")
    print(f"  lambda_: {lambda_.shape}")
    print(f"  x_prev: {x_prev.shape}")
    print(f"  realization: {realization.shape}")

    print(f"\nOutput predictions:")
    mode = config.integration["mode"]
    if mode == "subproblem":
        print(f"  subgradient shape: {predictions['subgradient'].shape}")
        print(f"  opt_value shape: {predictions['opt_value'].shape}")
        print(f"  opt_value: {predictions['opt_value']}")
    else:
        print(f"  x_init shape: {predictions['x_init'].shape}")
        print(f"  z_init shape: {predictions['z_init'].shape}")

    print("\n" + "=" * 60)
    print("Forward pass test passed!")
    print("=" * 60)


def test_cut_feature_builder():
    """测试 cut 特征构建"""
    print("\n" + "=" * 60)
    print("Testing Cut Feature Builder")
    print("=" * 60)

    num_cuts = 5
    gradient_dim = 13

    # 随机生成数据
    cut_gradients = torch.randn(num_cuts, gradient_dim)
    cut_intercepts = torch.randn(num_cuts)

    # 构建特征
    cut_features = NeuralWarmStartModel.build_cut_features(cut_gradients, cut_intercepts)

    print(f"\ncut_gradients shape: {cut_gradients.shape}")
    print(f"cut_intercepts shape: {cut_intercepts.shape}")
    print(f"cut_features shape: {cut_features.shape}")
    print(f"cut_features[:, -1] (phi_i): {cut_features[:, -1]}")

    print("\n" + "=" * 60)
    print("Cut feature builder test passed!")
    print("=" * 60)


def test_model_save_load():
    """测试模型保存和加载"""
    print("\n" + "=" * 60)
    print("Testing Model Save and Load")
    print("=" * 60)

    import tempfile
    from pathlib import Path

    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        temp_path = f.name

    try:
        # 创建并保存模型
        model = NeuralWarmStartModel()
        model.save(temp_path)
        print(f"Model saved to: {temp_path}")

        # 加载模型
        loaded_model = NeuralWarmStartModel.load(temp_path)
        loaded_model.eval()
        print("Model loaded successfully")

        # 测试加载的模型
        config = MLConfig()
        dims = config.dimensions
        cuts = torch.randn(1, 5, dims["cut_dim"])
        valid_mask = torch.ones(1, 5, dtype=torch.bool)
        lambda_ = torch.randn(1, dims["lambda_dim"])
        x_prev = torch.randn(1, dims["x_prev_dim"])
        realization = torch.randn(1, dims["realization_dim"])

        with torch.no_grad():
            predictions = loaded_model(
                cuts=cuts,
                valid_mask=valid_mask,
                lambda_=lambda_,
                x_prev=x_prev,
                realization=realization,
            )

        print(f"Loaded model prediction successful")
        if "subgradient" in predictions:
            print(f"  subgradient shape: {predictions['subgradient'].shape}")

    finally:
        # 清理临时文件
        import os
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            print(f"\nTemporary file deleted: {temp_path}")

    print("\n" + "=" * 60)
    print("Model save/load test passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_model_forward()
    test_cut_feature_builder()
    test_model_save_load()
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
