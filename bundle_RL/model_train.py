import csv
import os
from datetime import datetime
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# 自定义 Feature Extractor
class SimpleBundleExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)

        cuts_shape = observation_space["cuts"].shape
        pi_shape = observation_space["pi"].shape

        self.cuts_dim = cuts_shape[0] * cuts_shape[1]
        self.pi_dim = pi_shape[0]

        input_dim = self.cuts_dim + self.pi_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        cuts = observations["cuts"].view(observations["cuts"].shape[0], -1)
        pi = observations["pi"]

        x = torch.cat([cuts, pi], dim=1)
        return self.net(x)





def train(env, save_path=None, logger=None, model=None, total_timesteps=200_000, checkpoint_freq=5000, experiment_name="default", ent_coef=0):
    """
    训练函数
    
    Args:
        env: 环境
        save_path: 模型保存路径（可选）
        logger: 日志器
        model: 已有模型（用于继续训练）
        total_timesteps: 训练总步数
        checkpoint_freq: 检查点保存频率
        experiment_name: 实验名称，用于区分不同实验，训练结果将保存到 model/{experiment_name}/ 目录下
    """
    # 创建实验文件夹结构
    experiment_dir = os.path.join("model", experiment_name)
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    tensorboard_dir = os.path.join("res", "ppo_tensorboard", experiment_name)
    
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    # 统一提取超参数 (Hyperparameters)
    hparams = {
        "learning_rate": 3e-4,
        "n_steps": 512,  # 建议比 128 稍大，PPO 更稳定
        "batch_size": 128,
        "ent_coef": ent_coef,  # 开启微量探索
        "total_timesteps": total_timesteps,  # 训练总步数
        "features_dim": 128,  # 特征维度
        "net_arch": dict(pi=[128, 128], vf=[128, 128])  # 策略网络和价值网络结构
    }

    policy_kwargs = dict(
        features_extractor_class=SimpleBundleExtractor,
        features_extractor_kwargs=dict(features_dim=hparams["features_dim"]),
        net_arch=hparams["net_arch"]
    )

    # 如果没有传入模型，初始化新模型；否则继续训练已有模型
    if model is None:
        model = PPO(
            policy=MultiInputActorCriticPolicy,
            env=env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=hparams["learning_rate"],
            n_steps=hparams["n_steps"],
            batch_size=hparams["batch_size"],
            ent_coef=hparams["ent_coef"],
            tensorboard_log=tensorboard_dir  # TensorBoard 日志保存到实验目录
        )
    else:
        # 切换到新环境继续训练
        model.set_env(env)

    # 创建检查点回调（每 checkpoint_freq 步保存一次）
    from stable_baselines3.common.callbacks import CheckpointCallback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoints_dir,
        name_prefix="ppo_bundle_checkpoint"
    )

    model.learn(
        total_timesteps=hparams["total_timesteps"],
        reset_num_timesteps=False,
        callback=checkpoint_callback
    )

    # 保存路径
    timestamp = datetime.now().strftime("%m%d_%H%M")
    model_name = save_path if save_path else f"ppo_bundle_{timestamp}"
    final_save_path = os.path.join(experiment_dir, model_name)

    # 5. 保存模型
    model.save(final_save_path)

    # 4. 记录到 CSV 文件（每个实验独立的日志）
    csv_file = os.path.join(experiment_dir, "training_log.csv")

    # 准备这一行要存的数据
    row_data = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model_name": model_name,
        "experiment_name": experiment_name,
        **hparams  # 将 hparams 字典展开合并到 row_data
    }

    # 检查文件是否已存在，如果不存在则需要写表头 (Header)
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()  # 第一次创建文件时写入表头
        writer.writerow(row_data)

    msg = f"训练完成！实验: {experiment_name}, 模型: {model_name}.zip, 超参数已记录至: {csv_file}"
    if logger:
        logger.info(msg)
    else:
        print(msg)

    return model




