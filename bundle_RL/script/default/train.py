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


def get_experiment_dirs(experiment_name):
    """获取实验相关的目录路径"""
    # 获取当前文件所在目录的绝对路径，定位到 bundle_RL/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 上移两级目录：bundle_RL/script/default/ -> bundle_RL/script/ -> bundle_RL/
    base_dir = os.path.dirname(os.path.dirname(current_dir))
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
          resume=True, overwrite=False, learning_rate=3e-4, clip_range=0.2, clip_range_decay=True):
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
        ent_coef: 熵系数，控制探索程度
        resume: 是否自动从最新 checkpoint 继续训练（默认 True）
        overwrite: 是否覆盖已有模型重新训练（默认 False，即支持断点续训）
        learning_rate: 学习率（默认 3e-4）
        clip_range: PPO clip 范围（默认 0.2）
        clip_range_decay: 是否启用 clip_range 线性衰减（从 clip_range 衰减到 0.05）

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
        "learning_rate": learning_rate,
        "n_steps": 512,
        "batch_size": 128,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "clip_range_decay": clip_range_decay,
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

    # 使用可变对象存储 clip_range 值，以便在训练过程中可以修改
    current_clip_range = [clip_range]
    
    # clip_range 需要是可调用对象（函数）
    def clip_range_fn(_):
        return current_clip_range[0]
    
    if model is None:
        print(f"创建新模型，实验: {experiment_name}")
        print(f"超参数: learning_rate={learning_rate}, ent_coef={ent_coef}, clip_range={clip_range}")
        
        model = PPO(
            policy=MultiInputActorCriticPolicy,
            env=env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=learning_rate,
            n_steps=hparams["n_steps"],
            batch_size=hparams["batch_size"],
            ent_coef=ent_coef,
            clip_range=clip_range_fn,
            tensorboard_log=tensorboard_dir
        )
    else:
        print(f"继续训练已有模型，已训练步数: {existing_steps}")
        model.set_env(env)
        model.tensorboard_log = tensorboard_dir  # 恢复tensorboard日志配置
        model.learning_rate = learning_rate      # 更新学习率
        model.ent_coef = ent_coef                # 更新熵系数
        current_clip_range[0] = clip_range       # 更新clip范围（通过可变对象）
        model.clip_range = clip_range_fn         # 确保是可调用对象

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

    # 创建 clip_range 衰减回调
    callbacks = [checkpoint_callback]
    if clip_range_decay:
        from stable_baselines3.common.callbacks import BaseCallback
        
        class ClipRangeDecayCallback(BaseCallback):
            def __init__(self, current_clip_range, initial_clip_range, remaining_timesteps, total_timesteps, final_clip_range=0.05, verbose=0):
                super().__init__(verbose)
                self.current_clip_range = current_clip_range  # 可变列表引用
                self.initial_clip_range = initial_clip_range
                self.final_clip_range = final_clip_range
                self.remaining_timesteps = remaining_timesteps  # 剩余需要训练的步数
                self.total_timesteps = total_timesteps  # 总步数
                self.start_timesteps = total_timesteps - remaining_timesteps  # 续训前的累计步数
            
            def _on_step(self):
                # progress 应该基于本次续训的进度，而不是全局累计
                # self.num_timesteps 从 start_timesteps 开始累加
                current_progress = self.num_timesteps - self.start_timesteps
                progress = min(current_progress / self.remaining_timesteps, 1.0) if self.remaining_timesteps > 0 else 1.0
                # 通过可变对象修改 clip_range 值
                self.current_clip_range[0] = self.initial_clip_range - (self.initial_clip_range - self.final_clip_range) * progress
                return True
        
        clip_decay_callback = ClipRangeDecayCallback(current_clip_range, clip_range, remaining_timesteps, total_timesteps)
        callbacks.append(clip_decay_callback)
        print(f"启用 clip_range 衰减: 从 {clip_range} 线性衰减到 0.05")

    print(f"开始训练，剩余步数: {remaining_timesteps}/{total_timesteps}")
    if logger:
        logger.info(f"开始训练，剩余步数: {remaining_timesteps}/{total_timesteps}")

    print(f"model.policy: {model.policy}")
    print(f"model.policy.action_net: {model.policy.action_net}")
    print(f"model.policy.value_net: {model.policy.value_net}")

    model.learn(
        total_timesteps=remaining_timesteps,
        reset_num_timesteps=False,
        callback=callbacks,
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