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


import os
import json
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy


def train(env, save_path=None, logger=None):
    os.makedirs("model", exist_ok=True)

    # 统一提取超参数 (Hyperparameters)
    hparams = {
        "learning_rate": 3e-4,
        "n_steps": 512,  # 建议比 128 稍大，PPO 更稳定
        "batch_size": 128,
        "ent_coef": 0,  # 开启微量探索
        "total_timesteps": 200_000,  # 训练总步数
        "features_dim": 128,  # 特征维度
        "net_arch": dict(pi=[128, 128], vf=[128, 128])  # 策略网络和价值网络结构
    }

    policy_kwargs = dict(
        features_extractor_class=SimpleBundleExtractor,
        features_extractor_kwargs=dict(features_dim=hparams["features_dim"]),
        net_arch=hparams["net_arch"]
    )

    # 初始化并训练模型
    model = PPO(
        policy=MultiInputActorCriticPolicy,
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=hparams["learning_rate"],
        n_steps=hparams["n_steps"],
        batch_size=hparams["batch_size"],
        ent_coef=hparams["ent_coef"],
        tensorboard_log="./logs/ppo_tensorboard/"  # 训练数据会自动保存到这个文件夹
    )

    model.learn(total_timesteps=hparams["total_timesteps"])

    # 保存路径
    timestamp = datetime.now().strftime("%m%d_%H%M")
    model_name = save_path if save_path else f"ppo_bundle_{timestamp}"
    final_save_path = os.path.join("model", model_name)

    # 5. 保存模型
    model.save(final_save_path)

    # 4. 记录到 CSV 文件
    csv_file = os.path.join("model", "training_log.csv")

    # 准备这一行要存的数据
    row_data = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model_name": model_name,
        **hparams  # 将 hparams 字典展开合并到 row_data
    }

    # 检查文件是否已存在，如果不存在则需要写表头 (Header)
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()  # 第一次创建文件时写入表头
        writer.writerow(row_data)

    msg = f"训练完成！模型: {model_name}.zip, 超参数已记录至: {csv_file}"
    if logger:
        logger.info(msg)
    else:
        print(msg)

    return model




