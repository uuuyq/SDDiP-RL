"""
PPO 训练函数 for Level Bundle RL

遵循 bundle_RL/script/attention2/train.py 的代码风格:
- checkpoint 保存/加载
- training_log.csv 记录
- clip_range 衰减
- TensorBoard 日志
"""

import csv
import os
import re
from datetime import datetime

from stable_baselines3 import PPO

from bundle_norm_RL.script.features_extractor import LevelBundleFeaturesExtractor
from bundle_norm_RL.script.policy_network import LevelBundleActorCriticPolicy


def get_experiment_dirs(experiment_name):
    """获取实验相关的目录路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)  # bundle_norm_RL
    base_dir = str(base_dir)

    experiment_dir = os.path.join(base_dir, "train_result", "model", experiment_name)
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    tensorboard_dir = os.path.join(base_dir, "train_result", "ppo_tensorboard", experiment_name)
    save_dir = os.path.join(experiment_dir, "save")

    return experiment_dir, checkpoints_dir, tensorboard_dir, save_dir


def extract_steps_from_checkpoint(filename):
    match = re.search(r'_(\d+)_steps', filename)
    return int(match.group(1)) if match else 0


def load_latest_checkpoint(experiment_name, env, policy_kwargs):
    _, checkpoints_dir, _, _ = get_experiment_dirs(experiment_name)

    if not os.path.exists(checkpoints_dir):
        return None, 0

    checkpoints = [f for f in os.listdir(checkpoints_dir)
                   if f.startswith("ppo_level_bundle_checkpoint") and f.endswith(".zip")]

    if not checkpoints:
        return None, 0

    latest_checkpoint = max(checkpoints, key=lambda f: extract_steps_from_checkpoint(f))
    checkpoint_path = os.path.join(checkpoints_dir, latest_checkpoint)
    steps = extract_steps_from_checkpoint(latest_checkpoint)

    print(f"找到 checkpoint: {latest_checkpoint}, 已训练步数: {steps}")

    model = PPO.load(
        checkpoint_path,
        env=env,
        custom_objects={
            "LevelBundleFeaturesExtractor": LevelBundleFeaturesExtractor,
            "LevelBundleActorCriticPolicy": LevelBundleActorCriticPolicy,
            "policy_kwargs": policy_kwargs,
        }
    )

    return model, steps


def train(
    env,
    save_path=None,
    logger=None,
    model=None,
    total_timesteps=200_000,
    checkpoint_freq=5000,
    experiment_name="level_bundle_default",
    ent_coef=0,
    resume=True,
    overwrite=False,
    learning_rate=3e-4,
    clip_range=0.2,
    clip_range_decay=True,
    n_steps=512,
    batch_size=128,
    gamma=0.99,
    gae_lambda=0.95,
    n_epochs=10,
    vf_coef=0.5,
    max_grad_norm=0.5,
    target_kl=None,
    hidden_dim=64,
):
    """
    Level Bundle PPO 训练函数

    Returns:
        model, trained_steps, total_trained_steps
    """
    experiment_dir, checkpoints_dir, tensorboard_dir, save_dir = get_experiment_dirs(experiment_name)

    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    hparams = {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "clip_range_decay": clip_range_decay,
        "total_timesteps": total_timesteps,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "n_epochs": n_epochs,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "features_dim": 2 * hidden_dim,
        "hidden_dim": hidden_dim,
        "net_arch": dict(pi=[], vf=[]),
    }

    features_extractor_kwargs = dict(hidden_dim=hidden_dim)

    policy_kwargs = dict(
        features_extractor_class=LevelBundleFeaturesExtractor,
        features_extractor_kwargs=features_extractor_kwargs,
        net_arch=[],
    )

    existing_steps = 0

    if model is None and resume and not overwrite:
        model, existing_steps = load_latest_checkpoint(experiment_name, env, policy_kwargs)

    current_clip_range = [clip_range]

    def clip_range_fn(_):
        return current_clip_range[0]

    if model is None:
        print(f"创建新模型，实验: {experiment_name}")
        print(f"超参数: lr={learning_rate}, ent_coef={ent_coef}, clip_range={clip_range}")
        print(f"         n_steps={n_steps}, batch_size={batch_size}, gamma={gamma}")
        print(f"         hidden_dim={hidden_dim}, target_kl={target_kl}")

        ppo_kwargs = dict(
            policy=LevelBundleActorCriticPolicy,
            env=env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            n_epochs=n_epochs,
            ent_coef=ent_coef,
            clip_range=clip_range_fn,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            tensorboard_log=tensorboard_dir,
        )

        if target_kl is not None and isinstance(target_kl, (int, float)):
            ppo_kwargs['target_kl'] = target_kl

        model = PPO(**ppo_kwargs)
    else:
        print(f"继续训练已有模型，已训练步数: {existing_steps}")
        model.set_env(env)
        model.tensorboard_log = tensorboard_dir
        model.learning_rate = learning_rate
        model.ent_coef = ent_coef
        model.gamma = gamma
        model.gae_lambda = gae_lambda
        model.n_epochs = n_epochs
        model.vf_coef = vf_coef
        model.max_grad_norm = max_grad_norm
        if target_kl is not None and isinstance(target_kl, (int, float)):
            model.target_kl = target_kl
        current_clip_range[0] = clip_range
        model.clip_range = clip_range_fn

    if existing_steps >= total_timesteps:
        msg = f"模型已训练 {existing_steps} 步（目标 {total_timesteps} 步），无需继续训练"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        return model, 0, existing_steps

    remaining_timesteps = total_timesteps - existing_steps

    from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoints_dir,
        name_prefix="ppo_level_bundle_checkpoint",
    )

    callbacks = [checkpoint_callback]

    if clip_range_decay:
        class ClipRangeDecayCallback(BaseCallback):
            def __init__(self, current_clip_range, initial_clip_range, remaining_timesteps, total_timesteps,
                         final_clip_range=0.05, verbose=0, start_timesteps=0):
                super().__init__(verbose)
                self.current_clip_range = current_clip_range
                self.initial_clip_range = initial_clip_range
                self.final_clip_range = final_clip_range
                self.remaining_timesteps = remaining_timesteps
                self.start_timesteps = start_timesteps

            def _on_step(self):
                progress = min(
                    (self.num_timesteps - self.start_timesteps) / self.remaining_timesteps, 1.0
                ) if self.remaining_timesteps > 0 else 1.0
                self.current_clip_range[0] = (
                    self.initial_clip_range
                    - (self.initial_clip_range - self.final_clip_range) * progress
                )
                return True

        # 使用模型当前的 num_timesteps 作为衰减起点的全局步数，
        # 使交错训练中每个 config 都从初始 clip_range 开始独立衰减
        start_timesteps = model.num_timesteps if model is not None else 0
        callbacks.append(ClipRangeDecayCallback(
            current_clip_range, clip_range, remaining_timesteps, total_timesteps,
            start_timesteps=start_timesteps
        ))
        print(f"启用 clip_range 衰减: 从 {clip_range} 线性衰减到 0.05")

    print(f"开始训练，剩余步数: {remaining_timesteps}/{total_timesteps}")
    if logger:
        logger.info(f"开始训练，剩余步数: {remaining_timesteps}/{total_timesteps}")

    model.learn(
        total_timesteps=remaining_timesteps,
        reset_num_timesteps=False,
        callback=callbacks,
        tb_log_name="log",
    )

    total_trained_steps = existing_steps + remaining_timesteps

    timestamp = datetime.now().strftime("%m%d_%H%M")
    model_name = save_path if save_path else f"ppo_level_bundle_{timestamp}"
    final_save_path = os.path.join(save_dir, model_name)
    model.save(final_save_path)

    csv_file = os.path.join(experiment_dir, "training_log.csv")
    row_data = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model_name": model_name,
        "experiment_name": experiment_name,
        "trained_steps": total_trained_steps,
        "is_resumed": existing_steps > 0,
        **hparams,
    }

    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

    msg = (f"训练完成！实验: {experiment_name}, 模型: {model_name}.zip, "
           f"累计步数: {total_trained_steps}")
    if logger:
        logger.info(msg)
    else:
        print(msg)

    return model, remaining_timesteps, total_trained_steps
