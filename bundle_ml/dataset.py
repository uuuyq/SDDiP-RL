"""
数据处理模块：加载、清洗和整理训练数据
适配 bundle_ml.models.NeuralWarmStartModel

支持缓存功能：可将处理后的数据保存为 .pt 文件，后续直接加载缓存进行训练
"""

import hashlib
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# 缓存目录
CACHE_DIR = Path(__file__).parent / "cache"


def get_cache_path(data_dir: str, max_cuts: int = 50, normalize: bool = True) -> Path:
    """生成缓存文件路径"""
    # 使用数据目录的绝对路径生成哈希
    abs_data_dir = str(Path(data_dir).resolve())
    cache_key = hashlib.md5(f"{abs_data_dir}_{max_cuts}_{normalize}".encode()).hexdigest()[:8]

    # 获取数据目录名作为标识
    dir_name = Path(data_dir).name

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{dir_name}_{cache_key}.pt"


def save_cache(
    cache_path: Path,
    samples: List[Dict],
    dims: Dict[str, int],
    norm_params: Optional[Dict],
    instance_names: Optional[List[str]],
    max_cuts: int,
    normalize: bool,
):
    """保存数据缓存"""
    print(f"Saving cache to {cache_path}...")
    print(f"  Samples: {len(samples)}")
    print(f"  Dims: {dims}")

    # 转换为可序列化的格式，处理 None 值
    cache_data = {
        'samples': samples,
        'dims': dims,
        'norm_params': norm_params,
        'instance_names': instance_names,
        'max_cuts': max_cuts,
        'normalize': normalize,
    }

    try:
        torch.save(cache_data, cache_path)
        file_size = cache_path.stat().st_size
        print(f"Cache saved! {len(samples)} samples, size: {file_size / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"Failed to save cache: {e}")
        raise


def load_cache(
    cache_path: Path,
    instance_names: Optional[List[str]] = None,
    max_cuts: int = 50,
    normalize: bool = True,
) -> Optional[Tuple[List[Dict], Dict, Optional[Dict]]]:
    """加载数据缓存

    Returns:
        如果缓存有效且匹配，返回 (samples, dims, norm_params)
        否则返回 None
    """
    if not cache_path.exists():
        return None

    try:
        cache_data = torch.load(cache_path, map_location='cpu', weights_only=False)

        # 检查参数是否匹配
        if cache_data.get('max_cuts') != max_cuts:
            print(f"Cache max_cuts mismatch: {cache_data.get('max_cuts')} != {max_cuts}")
            return None
        if cache_data.get('normalize') != normalize:
            print(f"Cache normalize mismatch: {cache_data.get('normalize')} != {normalize}")
            return None

        # 检查实例名是否匹配（如果指定了的话）
        cached_instances = cache_data.get('instance_names')
        if instance_names is not None and cached_instances is not None:
            # 如果都指定了，检查是否一致
            if set(instance_names) != set(cached_instances):
                print(f"Cache instance_names mismatch")
                return None

        print(f"Loaded cache from {cache_path}: {len(cache_data['samples'])} samples")
        return (
            cache_data['samples'],
            cache_data['dims'],
            cache_data['norm_params'],
        )
    except Exception as e:
        print(f"Failed to load cache: {e}")
        return None


class BundleDataset(Dataset):
    """
    Bundle Method 数据集

    适配 NeuralWarmStartModel 的输入格式:
    - cuts: (num_cuts, cut_dim) - 变长，需要 padding
    - valid_mask: (num_cuts,) - 有效 cut 的 mask
    - lambda_: (lambda_dim,)
    - x_prev: (x_prev_dim,) - 来自 trial_point
    - realization: (realization_dim,) - p_d + re
    - stage: int

    支持缓存功能：如果缓存存在则直接加载，否则从原始数据加载并生成缓存
    """

    def __init__(
        self,
        data_dir: str,
        instance_names: Optional[List[str]] = None,
        max_samples_per_file: Optional[int] = None,
        max_cuts: int = 50,
        normalize: bool = True,
        use_cache: bool = True,
    ):
        """
        Args:
            data_dir: 数据目录路径
            instance_names: 要加载的实例名列表，None 表示全部
            max_samples_per_file: 每个文件最多采样数量（用于快速测试）
            max_cuts: 最大 cuts 数量（用于 padding）
            normalize: 是否进行归一化
            use_cache: 是否使用缓存（默认开启）
        """
        self.data_dir = Path(data_dir)
        self.max_samples_per_file = max_samples_per_file
        self.max_cuts = max_cuts
        self.normalize = normalize
        self.instance_names = instance_names

        # 尝试从缓存加载
        cache_loaded = False
        if use_cache:
            cache_path = get_cache_path(data_dir, max_cuts, normalize)
            cache_result = load_cache(
                cache_path,
                instance_names=instance_names,
                max_cuts=max_cuts,
                normalize=normalize,
            )
            if cache_result is not None:
                self.samples, dims, self.norm_params = cache_result
                self.lambda_dim = dims['lambda_dim']
                self.x_prev_dim = dims['x_prev_dim']
                self.realization_dim = dims['realization_dim']
                self.cut_dim = dims['cut_dim']
                self.output_dim = dims['output_dim']
                self.max_cuts = dims.get('max_cuts', max_cuts)
                cache_loaded = True
                print(f"Using cached data: {len(self.samples)} samples")

        if not cache_loaded:
            # 收集所有数据
            self.samples = []
            self._load_data(instance_names)

            # 解析维度
            self._parse_dimensions()

            # 计算归一化参数
            if self.normalize and self.samples:
                self._compute_normalization_params()
            else:
                self.norm_params = None

            # 保存缓存
            if use_cache and self.samples:
                cache_path = get_cache_path(data_dir, max_cuts, normalize)
                dims = self.get_dimensions()
                save_cache(
                    cache_path=cache_path,
                    samples=self.samples,
                    dims=dims,
                    norm_params=self.norm_params,
                    instance_names=instance_names,
                    max_cuts=max_cuts,
                    normalize=normalize,
                )
    
    def _load_data(self, instance_names: Optional[List[str]]):
        """加载所有 JSON 数据文件"""
        if instance_names is None:
            # 加载所有子目录
            instance_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        else:
            instance_dirs = [self.data_dir / name for name in instance_names if (self.data_dir / name).is_dir()]
        
        for instance_dir in instance_dirs:
            json_files = list(instance_dir.glob("*.json"))
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 如果设置了 max_samples_per_file，随机采样
                    if self.max_samples_per_file and len(data) > self.max_samples_per_file:
                        indices = np.random.choice(len(data), self.max_samples_per_file, replace=False)
                        data = [data[i] for i in indices]
                    
                    self.samples.extend(data)
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
        
        print(f"Loaded {len(self.samples)} samples from {len(instance_dirs)} instances")
    
    def _parse_dimensions(self):
        """解析数据维度"""
        if not self.samples:
            self.lambda_dim = 13
            self.x_prev_dim = 13
            self.realization_dim = 12
            self.cut_dim = 14
            self.output_dim = 13
            return
        
        sample = self.samples[0]
        
        # Lambda 维度
        self.lambda_dim = len(sample['lambda'])
        
        # x_prev 维度 (trial_point 展开)
        # trial_point = X_TRIAL + Y_TRIAL + X_BS_TRIAL(展开) + SOC_TRIAL
        x_trial = sample['X_TRIAL'] if sample.get('X_TRIAL') else []
        y_trial = sample['Y_TRIAL'] if sample.get('Y_TRIAL') else []
        x_bs_trial = sample['X_BS_TRIAL'] if sample.get('X_BS_TRIAL') else []
        soc_trial = sample['SOC_TRIAL'] if sample.get('SOC_TRIAL') else []
        
        # 展开 x_bs_trial
        x_bs_flat = []
        for bs in x_bs_trial:
            if isinstance(bs, list):
                x_bs_flat.extend(bs)
            else:
                x_bs_flat.append(bs)
        
        self.x_prev_dim = len(x_trial) + len(y_trial) + len(x_bs_flat) + len(soc_trial)
        
        # Realization 维度
        p_d = sample['p_d'] if sample.get('p_d') else []
        re = sample['re'] if sample.get('re') else []
        self.realization_dim = len(p_d) + len(re)
        
        # Cut 维度 (subgradient + intercept)
        subgradient = sample['subgradient'] if sample.get('subgradient') else []
        self.cut_dim = len(subgradient) + 1  # subgradient + intercept
        
        # Output 维度 (subgradient)
        self.output_dim = len(subgradient)
        
        print(f"Parsed dimensions:")
        print(f"  lambda_dim: {self.lambda_dim}")
        print(f"  x_prev_dim: {self.x_prev_dim}")
        print(f"  realization_dim: {self.realization_dim}")
        print(f"  cut_dim: {self.cut_dim}")
        print(f"  output_dim: {self.output_dim}")
    
    def _compute_normalization_params(self):
        """计算归一化参数"""
        self.norm_params = {
            'lambda': {'mean': None, 'std': None},
            'x_prev': {'mean': None, 'std': None},
            'realization': {'mean': None, 'std': None},
            'subgradient': {'mean': None, 'std': None},
        }
        
        # 收集 lambda
        lambda_values = []
        for sample in self.samples:
            lambda_values.extend(sample['lambda'])
        
        if lambda_values:
            lambda_values = np.array(lambda_values)
            self.norm_params['lambda']['mean'] = float(np.mean(lambda_values))
            self.norm_params['lambda']['std'] = float(np.std(lambda_values)) + 1e-8
        
        # 收集 x_prev
        for sample in self.samples:
            x_prev = self._get_x_prev(sample)
            if not hasattr(self, '_x_prev_values'):
                self._x_prev_values = []
            self._x_prev_values.extend(x_prev)
        
        if hasattr(self, '_x_prev_values') and self._x_prev_values:
            x_prev_arr = np.array(self._x_prev_values)
            self.norm_params['x_prev']['mean'] = float(np.mean(x_prev_arr))
            self.norm_params['x_prev']['std'] = float(np.std(x_prev_arr)) + 1e-8
            del self._x_prev_values
        
        # 收集 realization
        real_values = []
        for sample in self.samples:
            if sample.get('p_d'):
                real_values.extend(sample['p_d'])
            if sample.get('re'):
                real_values.extend(sample['re'])
        
        if real_values:
            real_arr = np.array(real_values)
            self.norm_params['realization']['mean'] = float(np.mean(real_arr))
            self.norm_params['realization']['std'] = float(np.std(real_arr)) + 1e-8
        
        # 收集 subgradient
        sg_values = []
        for sample in self.samples:
            sg_values.extend(sample['subgradient'])
        
        if sg_values:
            sg_arr = np.array(sg_values)
            self.norm_params['subgradient']['mean'] = float(np.mean(sg_arr))
            self.norm_params['subgradient']['std'] = float(np.std(sg_arr)) + 1e-8
        
        print("Normalization parameters computed:")
        for key, params in self.norm_params.items():
            if params['mean'] is not None:
                print(f"  {key}: mean={params['mean']:.4f}, std={params['std']:.4f}")
    
    def _normalize(self, key: str, values: np.ndarray) -> np.ndarray:
        """对特征进行归一化"""
        if not self.normalize or self.norm_params is None:
            return values
        
        params = self.norm_params.get(key)
        if params is None or params['mean'] is None:
            return values
        
        # 添加保护，防止除以接近 0 的值
        std = params['std'] if params['std'] > 1e-8 else 1e-8
        return (values - params['mean']) / std
    
    def _get_x_prev(self, sample: Dict) -> List[float]:
        """从样本中提取 x_prev (trial_point)"""
        x_trial = sample.get('X_TRIAL', [])
        y_trial = sample.get('Y_TRIAL', [])
        x_bs_trial = sample.get('X_BS_TRIAL', [])
        soc_trial = sample.get('SOC_TRIAL', [])
        
        # 展开 x_bs_trial
        x_bs_flat = []
        for bs in x_bs_trial:
            if isinstance(bs, list):
                x_bs_flat.extend(bs)
            else:
                x_bs_flat.append(bs)
        
        return x_trial + y_trial + x_bs_flat + soc_trial
    
    def _pad_cuts(self, cuts: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Padding cuts 到固定长度
        
        Returns:
            cuts_padded: (max_cuts, cut_dim), padding 值为 0
            valid_mask: (max_cuts,), True 表示有效
        """
        if not cuts:
            cuts_padded = np.zeros((self.max_cuts, self.cut_dim), dtype=np.float32)
            valid_mask = np.zeros(self.max_cuts, dtype=bool)
            return cuts_padded, valid_mask
        
        num_cuts = min(len(cuts), self.max_cuts)
        cuts_padded = np.zeros((self.max_cuts, self.cut_dim), dtype=np.float32)
        valid_mask = np.zeros(self.max_cuts, dtype=bool)
        
        for i in range(num_cuts):
            cut = cuts[i]
            # 填充到 cut_dim
            if len(cut) < self.cut_dim:
                cut = cut + [0.0] * (self.cut_dim - len(cut))
            elif len(cut) > self.cut_dim:
                cut = cut[:self.cut_dim]
            cuts_padded[i] = cut
            valid_mask[i] = True
        
        return cuts_padded, valid_mask
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        返回一个样本
        
        Returns:
            dict:
                - cuts: (max_cuts, cut_dim)
                - valid_mask: (max_cuts,)
                - lambda_: (lambda_dim,)
                - x_prev: (x_prev_dim,)
                - realization: (realization_dim,)
                - stage: int
                - subgradient: (subgradient_dim,)
                - opt_value: (1,)
        """
        sample = self.samples[idx]
        
        # 提取特征
        lambda_ = np.array(sample['lambda'], dtype=np.float32)
        x_prev = np.array(self._get_x_prev(sample), dtype=np.float32)
        
        # Realization
        p_d = np.array(sample.get('p_d', []), dtype=np.float32)
        re = np.array(sample.get('re', []), dtype=np.float32)
        realization = np.concatenate([p_d, re]) if len(p_d) > 0 or len(re) > 0 else np.array([], dtype=np.float32)
        
        # Cuts
        lag_cuts = sample.get('lag_cuts', [])
        cuts, valid_mask = self._pad_cuts(lag_cuts)
        
        # Stage
        stage = sample.get('stage', 0)
        
        # 目标
        subgradient = np.array(sample['subgradient'], dtype=np.float32)
        opt_value = np.array([sample['opt_value']], dtype=np.float32)
        
        # 归一化
        lambda_ = self._normalize('lambda', lambda_)
        x_prev = self._normalize('x_prev', x_prev)
        if len(realization) > 0:
            realization = self._normalize('realization', realization)
        
        # 确保 realization 维度正确
        if len(realization) == 0:
            realization = np.zeros(self.realization_dim, dtype=np.float32)
        
        return {
            'cuts': torch.from_numpy(cuts),
            'valid_mask': torch.from_numpy(valid_mask),
            'lambda_': torch.from_numpy(lambda_),
            'x_prev': torch.from_numpy(x_prev),
            'realization': torch.from_numpy(realization),
            'stage': torch.tensor(stage, dtype=torch.long),
            'subgradient': torch.from_numpy(subgradient),
            'opt_value': torch.from_numpy(opt_value),
        }
    
    def get_dimensions(self) -> Dict[str, int]:
        """获取维度信息"""
        return {
            'lambda_dim': self.lambda_dim,
            'x_prev_dim': self.x_prev_dim,
            'realization_dim': self.realization_dim,
            'cut_dim': self.cut_dim,
            'output_dim': self.output_dim,
            'max_cuts': self.max_cuts,
        }


class DataCollator:
    """数据整理器，将多个样本整理成批次"""
    
    def __init__(self, max_cuts: int = 50, pad_value: float = 0.0):
        self.max_cuts = max_cuts
        self.pad_value = pad_value
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """整理批次数据"""
        return {
            'cuts': torch.stack([item['cuts'] for item in batch]),
            'valid_mask': torch.stack([item['valid_mask'] for item in batch]),
            'lambda_': torch.stack([item['lambda_'] for item in batch]),
            'x_prev': torch.stack([item['x_prev'] for item in batch]),
            'realization': torch.stack([item['realization'] for item in batch]),
            'stage': torch.stack([item['stage'] for item in batch]),
            'subgradient': torch.stack([item['subgradient'] for item in batch]),
            'opt_value': torch.stack([item['opt_value'] for item in batch]),
        }


def analyze_data(data_dir: str) -> Dict:
    """分析数据分布"""
    dataset = BundleDataset(data_dir, normalize=False)
    
    stats = {
        'num_samples': len(dataset),
        'dimensions': dataset.get_dimensions(),
    }
    
    return stats


if __name__ == '__main__':
    # 测试数据加载
    data_dir = r"D:\tools\workspace_pycharm\SDDiP-RL\bundle_ml\training_data"
    stats = analyze_data(data_dir)
    print("\nData Analysis:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
