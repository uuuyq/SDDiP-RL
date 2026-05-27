"""
训练脚本：训练 Neural Warm Start 模型
"""

import argparse
import os
import yaml
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from bundle_ml.dataset import BundleDataset, DataCollator
from bundle_ml.models import NeuralWarmStartModel
from bundle_ml.ml_config import MLConfig


class BundleLoss(nn.Module):
    """Bundle Method 损失函数"""
    
    def __init__(self, subgradient_weight: float = 1.0, opt_value_weight: float = 0.1):
        super().__init__()
        self.subgradient_weight = subgradient_weight
        self.opt_value_weight = opt_value_weight
        self.mse = nn.MSELoss()
    
    def forward(
        self,
        pred_subgradient: torch.Tensor,
        pred_opt_value: torch.Tensor,
        target_subgradient: torch.Tensor,
        target_opt_value: torch.Tensor,
    ) -> tuple:
        loss_subgradient = self.mse(pred_subgradient, target_subgradient)
        loss_opt_value = self.mse(pred_opt_value, target_opt_value)
        loss = self.subgradient_weight * loss_subgradient + self.opt_value_weight * loss_opt_value
        return loss, {'loss_subgradient': loss_subgradient.item(), 'loss_opt_value': loss_opt_value.item()}


def create_model_from_data(dataset: BundleDataset, config_path: str = None) -> NeuralWarmStartModel:
    """
    根据数据集维度创建模型
    
    Args:
        dataset: BundleDataset 对象
        config_path: 配置文件路径，如果为 None 则使用默认配置并更新维度
        
    Returns:
        model: NeuralWarmStartModel
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yml"
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 更新维度
    dims = config_dict['dimensions']
    dims['lambda_dim'] = dataset.lambda_dim
    dims['x_prev_dim'] = dataset.x_prev_dim
    dims['realization_dim'] = dataset.realization_dim
    dims['cut_dim'] = dataset.cut_dim
    dims['output_dim'] = dataset.output_dim
    
    # 创建配置对象
    class DynamicConfig:
        def __init__(self, d):
            self._config = d
        
        @property
        def dimensions(self):
            return self._config['dimensions']
        
        @property
        def network(self):
            return self._config['network']
        
        @property
        def integration(self):
            return self._config['integration']
        
        @property
        def data(self):
            return self._config['data']
    
    model_config = DynamicConfig(config_dict)
    return NeuralWarmStartModel(config=model_config)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: BundleLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict:
    """训练一个 epoch"""
    model.train()
    
    total_loss = 0
    total_sg_loss = 0
    total_ov_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        # 准备输入
        cuts = batch['cuts'].to(device)
        valid_mask = batch['valid_mask'].to(device)
        lambda_ = batch['lambda_'].to(device)
        x_prev = batch['x_prev'].to(device)
        realization = batch['realization'].to(device)
        stage = batch['stage'].to(device)
        
        target_subgradient = batch['subgradient'].to(device)
        target_opt_value = batch['opt_value'].to(device)
        
        # 前向传播
        predictions = model(
            cuts=cuts,
            valid_mask=valid_mask,
            lambda_=lambda_,
            x_prev=x_prev,
            realization=realization,
            stage=stage,
        )
        
        pred_subgradient = predictions['subgradient']
        pred_opt_value = predictions['opt_value']
        
        # 计算损失
        loss, loss_dict = criterion(
            pred_subgradient, pred_opt_value,
            target_subgradient, target_opt_value
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_sg_loss += loss_dict['loss_subgradient']
        total_ov_loss += loss_dict['loss_opt_value']
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'loss_subgradient': total_sg_loss / num_batches,
        'loss_opt_value': total_ov_loss / num_batches,
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: BundleLoss,
    device: torch.device,
) -> dict:
    """验证模型"""
    model.eval()
    
    total_loss = 0
    total_sg_loss = 0
    total_ov_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            cuts = batch['cuts'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            lambda_ = batch['lambda_'].to(device)
            x_prev = batch['x_prev'].to(device)
            realization = batch['realization'].to(device)
            stage = batch['stage'].to(device)
            
            target_subgradient = batch['subgradient'].to(device)
            target_opt_value = batch['opt_value'].to(device)
            
            predictions = model(
                cuts=cuts,
                valid_mask=valid_mask,
                lambda_=lambda_,
                x_prev=x_prev,
                realization=realization,
                stage=stage,
            )
            
            pred_subgradient = predictions['subgradient']
            pred_opt_value = predictions['opt_value']
            
            loss, loss_dict = criterion(
                pred_subgradient, pred_opt_value,
                target_subgradient, target_opt_value
            )
            
            total_loss += loss.item()
            total_sg_loss += loss_dict['loss_subgradient']
            total_ov_loss += loss_dict['loss_opt_value']
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'loss_subgradient': total_sg_loss / num_batches,
        'loss_opt_value': total_ov_loss / num_batches,
    }


def train(args):
    """主训练函数"""
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    print(f"\nLoading data from: {args.data_dir}")
    dataset = BundleDataset(
        data_dir=args.data_dir,
        instance_names=args.instance_names,
        max_samples_per_file=args.max_samples_per_file,
        max_cuts=args.max_cuts,
        normalize=True,
    )
    
    print(f"Total samples: {len(dataset)}")
    dims = dataset.get_dimensions()
    print(f"Dimensions: {dims}")
    
    # 划分训练集和验证集
    val_ratio = args.val_ratio
    train_size = int(len(dataset) * (1 - val_ratio))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # 创建 DataLoader
    collator = DataCollator(max_cuts=args.max_cuts)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )
    
    # 创建模型
    model = create_model_from_data(dataset).to(device)
    
    print(f"\nModel created:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = BundleLoss(
        subgradient_weight=args.subgradient_weight,
        opt_value_weight=args.opt_value_weight,
    )
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
    )
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"train_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=str(output_dir / 'logs'))
    
    # 保存配置
    config_save = {
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'subgradient_weight': args.subgradient_weight,
        'opt_value_weight': args.opt_value_weight,
        'dimensions': dims,
    }
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config_save, f)
    
    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Output directory: {output_dir}")
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, device)
        
        # 学习率调度
        scheduler.step(val_metrics['loss'])
        
        # 记录到 TensorBoard
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Loss_subgradient/train', train_metrics['loss_subgradient'], epoch)
        writer.add_scalar('Loss_subgradient/val', val_metrics['loss_subgradient'], epoch)
        writer.add_scalar('Loss_opt_value/train', train_metrics['loss_opt_value'], epoch)
        writer.add_scalar('Loss_opt_value/val', val_metrics['loss_opt_value'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # 打印
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  Train - Loss: {train_metrics['loss']:.6f}, "
              f"Subgradient: {train_metrics['loss_subgradient']:.6f}, "
              f"Opt_value: {train_metrics['loss_opt_value']:.6f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.6f}, "
              f"Subgradient: {val_metrics['loss_subgradient']:.6f}, "
              f"Opt_value: {val_metrics['loss_opt_value']:.6f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # 保存最佳模型
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'dimensions': dims,
            }, output_dir / 'best_model.pt')
            
            print(f"  -> Best model saved!")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    writer.close()
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Output directory: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train Neural Warm Start Model')
    
    # 数据配置
    parser.add_argument('--data_dir', type=str,
                        default=r"D:\tools\workspace_pycharm\SDDiP-RL\bundle_ml\training_data",
                        help='数据目录路径')
    parser.add_argument('--instance_names', type=str, nargs='+', default=None,
                        help='要加载的实例名列表')
    parser.add_argument('--max_samples_per_file', type=int, default=None,
                        help='每个文件最多采样数量（用于快速测试）')
    parser.add_argument('--max_cuts', type=int, default=50,
                        help='最大 cuts 数量')
    
    # 训练配置
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='权重衰减')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='验证集比例')
    parser.add_argument('--patience', type=int, default=10,
                        help='早停耐心值')
    
    # 损失配置
    parser.add_argument('--subgradient_weight', type=float, default=1.0,
                        help='子梯度损失权重')
    parser.add_argument('--opt_value_weight', type=float, default=0.1,
                        help='最优值损失权重')
    
    # 其他配置
    parser.add_argument('--output_dir', type=str,
                        default=r"D:\tools\workspace_pycharm\SDDiP-RL\bundle_ml\checkpoints",
                        help='输出目录')
    parser.add_argument('--use_cuda', action='store_true',
                        help='是否使用 CUDA')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
