import csv
import os
import re
from datetime import datetime
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SimpleBundleExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)

        cuts_shape = observation_space["cuts"].shape
        pi_shape = observation_space["pi"].shape
        trial_point_shape = observation_space["trial_point"].shape
        realization_shape = observation_space["realization"].shape

        self.cuts_dim = cuts_shape[0] * cuts_shape[1]
        self.pi_dim = pi_shape[0]
        self.trial_point_dim = trial_point_shape[0]
        self.realization_dim = realization_shape[0]

        input_dim = self.cuts_dim + self.pi_dim + self.trial_point_dim + self.realization_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        cuts = observations["cuts"].view(observations["cuts"].shape[0], -1)
        pi = observations["pi"]
        trial_point = observations["trial_point"]
        realization = observations["realization"]

        x = torch.cat([cuts, pi, trial_point, realization], dim=1)
        return self.net(x)


def get_experiment_dirs(experiment_name):
    """获取实验相关的目录路径"""
    # 获取当前文件所在目录的绝对路径，定位到 bundle_RL/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)  # 上一级目录 bundle_RL/script/ -> bundle_RL/
    
    # 确保 base_dir 是 str 类型，避免类型检查警告
    base_dir = str(base_dir)
    
    experiment_dir = os.path.join(base_dir, "train_result", "model", experiment_name)
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    tensorboard_dir = os.path.join(base_dir, "train_result", "ppo_tensorboard", experiment_name)
    save_dir = os.path.join(experiment_dir, "save")

    return experiment_dir, checkpoints_dir, tensorboard_dir, save_dir


def extract_steps_from_checkpoint(filename):
    """从 checkpoint 文件名中提取步数"""
    match = re.search(r'_(\d+)_steps', filename)
    if match:
        return int(match.group(1))
    return 0


def load_latest_checkpoint(experiment_name, env):
    """
    加载最新的 checkpoint（如果存在）

    Args:
        experiment_name: 实验名称
        env: 环境（用于加载模型）

    Returns:
        model: 加载的模型，如果不存在则返回 None
        steps: 已训练的步数，如果不存在则返回 0
    """
    _, checkpoints_dir, tensorboard_dir, _ = get_experiment_dirs(experiment_name)

    if not os.path.exists(checkpoints_dir):
        return None, 0

    checkpoints = [f for f in os.listdir(checkpoints_dir)
                   if f.startswith("ppo_bundle_checkpoint") and f.endswith(".zip")]

    if not checkpoints:
        return None, 0

    latest_checkpoint = max(checkpoints, key=lambda f: extract_steps_from_checkpoint(f))
    checkpoint_path = os.path.join(checkpoints_dir, latest_checkpoint)
    steps = extract_steps_from_checkpoint(latest_checkpoint)

    print(f"找到 checkpoint: {latest_checkpoint}, 已训练步数: {steps}")

    policy_kwargs = dict(
        features_extractor_class=SimpleBundleExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )

    model = PPO.load(
        checkpoint_path,
        env=env,
        custom_objects={
            "SimpleBundleExtractor": SimpleBundleExtractor,
            "policy_kwargs": policy_kwargs
        }
    )

    return model, steps


def train(env, save_path=None, logger=None, model=None, total_timesteps=200_000,
          checkpoint_freq=5000, experiment_name="default", ent_coef=0,
          resume=True, overwrite=False):
    """
    训练函数，支持断点续训

    Args:
        env: 环境
        save_path: 模型保存路径（可选）
        logger: 日志器
        model: 已有模型（用于继续训练）
        total_timesteps: 训练总步数
        checkpoint_freq: 检查点保存频率
        experiment_name: 实验名称，用于区分不同实验
        ent_coef: 探索系数
        resume: 是否自动从最新 checkpoint 继续训练（默认 True）
        overwrite: 是否覆盖已有模型重新训练（默认 False，即支持断点续训）

    Returns:
        model: 训练后的模型
        trained_steps: 本次训练的步数
        total_trained_steps: 累计训练的步数
    """
    experiment_dir, checkpoints_dir, tensorboard_dir, save_dir = get_experiment_dirs(experiment_name)

    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    hparams = {
        "learning_rate": 3e-4,
        "n_steps": 512,
        "batch_size": 128,
        "ent_coef": ent_coef,
        "total_timesteps": total_timesteps,
        "features_dim": 128,
        "net_arch": dict(pi=[128, 128], vf=[128, 128])
    }

    policy_kwargs = dict(
        features_extractor_class=SimpleBundleExtractor,
        features_extractor_kwargs=dict(features_dim=hparams["features_dim"]),
        net_arch=hparams["net_arch"]
    )

    loaded_from_checkpoint = False
    existing_steps = 0

    if model is None and resume and not overwrite:
        model, existing_steps = load_latest_checkpoint(experiment_name, env)

    if model is None:
        print(f"创建新模型，实验: {experiment_name}")
        model = PPO(
            policy=MultiInputActorCriticPolicy,
            env=env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=hparams["learning_rate"],
            n_steps=hparams["n_steps"],
            batch_size=hparams["batch_size"],
            ent_coef=hparams["ent_coef"],
            tensorboard_log=tensorboard_dir
        )
    else:
        print(f"继续训练已有模型，已训练步数: {existing_steps}")
        model.set_env(env)
        model.tensorboard_log = tensorboard_dir  # 恢复tensorboard日志配置

    if existing_steps >= total_timesteps:
        msg = f"模型已训练 {existing_steps} 步（目标 {total_timesteps} 步），无需继续训练"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        return model, 0, existing_steps

    remaining_timesteps = total_timesteps - existing_steps

    from stable_baselines3.common.callbacks import CheckpointCallback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoints_dir,
        name_prefix="ppo_bundle_checkpoint"
    )

    print(f"开始训练，剩余步数: {remaining_timesteps}/{total_timesteps}")
    if logger:
        logger.info(f"开始训练，剩余步数: {remaining_timesteps}/{total_timesteps}")

    print(f"model.policy: {model.policy}")
    print(f"model.policy.action_net: {model.policy.action_net}")
    print(f"model.policy.value_net: {model.policy.value_net}")

    model.learn(
        total_timesteps=remaining_timesteps,
        reset_num_timesteps=False,
        callback=checkpoint_callback,
        tb_log_name="log"  # 使用固定名称，避免在tensorboard_dir下创建额外子目录
    )

    total_trained_steps = existing_steps + remaining_timesteps

    timestamp = datetime.now().strftime("%m%d_%H%M")
    model_name = save_path if save_path else f"ppo_bundle_{timestamp}"
    final_save_path = os.path.join(save_dir, model_name)

    model.save(final_save_path)

    csv_file = os.path.join(experiment_dir, "training_log.csv")

    row_data = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model_name": model_name,
        "experiment_name": experiment_name,
        "trained_steps": total_trained_steps,
        "is_resumed": existing_steps > 0,
        **hparams
    }

    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

    msg = (f"训练完成！实验: {experiment_name}, 模型: {model_name}.zip, "
           f"累计步数: {total_trained_steps}, 超参数已记录至: {csv_file}")
    if logger:
        logger.info(msg)
    else:
        print(msg)

    return model, remaining_timesteps, total_trained_steps