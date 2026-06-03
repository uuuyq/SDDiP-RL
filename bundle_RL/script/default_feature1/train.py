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
        search_dir_norm_shape = observation_space["search_direction_norm"].shape
        linear_imp_shape = observation_space["linear_improvement"].shape
        cosine_sim_shape = observation_space["cosine_sim"].shape

        self.cuts_dim = cuts_shape[0] * cuts_shape[1]
        self.pi_dim = pi_shape[0]
        self.trial_point_dim = trial_point_shape[0]
        self.realization_dim = realization_shape[0]
        self.search_dir_norm_dim = search_dir_norm_shape[0]
        self.linear_imp_dim = linear_imp_shape[0]
        self.cosine_sim_dim = cosine_sim_shape[0]

        # 计算处理后的输入维度
        input_dim = (self.pi_dim + self.cosine_sim_dim) + \
                    (self.linear_imp_dim + self.search_dir_norm_dim) + \
                    (self.trial_point_dim + self.realization_dim) + \
                    self.cuts_dim

        # 在网络入口处添加 LayerNorm
        self.input_ln = nn.LayerNorm(input_dim)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, features_dim),
            nn.GELU(),
        )

    def forward(self, observations):
        cuts = observations["cuts"].view(observations["cuts"].shape[0], -1)
        pi = observations["pi"]
        trial_point = observations["trial_point"]
        realization = observations["realization"]
        search_dir_norm = observations["search_direction_norm"]
        linear_imp = observations["linear_improvement"]
        cosine_sim = observations["cosine_sim"]

        # 1. 纯天然健康特征，保持原样
        feat_pure = torch.cat([pi, cosine_sim], dim=-1)
        
        # 2. 对数转换（治愈极端大数）
        linear_imp_log = torch.log(linear_imp + 1.0)
        search_dir_norm_log = torch.log(search_dir_norm + 1.0)
        feat_log = torch.cat([linear_imp_log, search_dir_norm_log], dim=-1)
        
        # 3. 坐标/物理量特征（基础缩放）
        feat_scale = torch.cat([trial_point, realization], dim=-1) / 100.0
        
        # 4. 割平面特征
        feat_cuts = cuts
        
        # 5. 最终大拼接
        state_vector = torch.cat([feat_pure, feat_log, feat_scale, feat_cuts], dim=-1)
        
        # 6. 全局 LayerNorm 做最终的协同对齐
        state_vector = self.input_ln(state_vector)

        return self.net(state_vector)


def get_experiment_dirs(experiment_name):
    """获取实验相关的目录路径"""
    # 获取当前文件所在目录的绝对路径，定位到 bundle_RL/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 上移两级目录：bundle_RL/script/default_feature/ -> bundle_RL/script/ -> bundle_RL/
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


def load_latest_checkpoint(experiment_name, env, policy_kwargs):
    """
    加载最新的 checkpoint（如果存在）

    Args:
        experiment_name: 实验名称
        env: 环境（用于加载模型）
        policy_kwargs: 策略参数

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
          resume=True, overwrite=False, learning_rate=3e-4, clip_range=0.2, clip_range_decay=True,
          n_steps=512, batch_size=128, gamma=0.99, gae_lambda=0.95, n_epochs=10,
          vf_coef=0.5, max_grad_norm=0.5, target_kl=None,
          features_dim=128, hidden_dim=64, num_heads=4, num_layers=1, ffn_dim=128, dropout=0.1,
          actor_net_arch=None, critic_net_arch=None, share_encoder=True):
    """
    训练函数，支持断点续训，接受与 attention1 一致的参数签名

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
        n_steps: 每次更新采集的步数
        batch_size: 批大小
        gamma: 折扣因子
        gae_lambda: GAE 参数
        n_epochs: 训练轮数
        vf_coef: 价值函数系数
        max_grad_norm: 最大梯度范数
        target_kl: KL 散度目标（用于提前终止训练轮次）
        features_dim: 特征提取器维度（本模块固定 128）
        hidden_dim: 编码器隐藏层维度（本模块不使用）
        num_heads: Attention 头数（本模块不使用）
        num_layers: Attention 层数（本模块不使用）
        ffn_dim: FFN 维度（本模块不使用）
        dropout: Dropout 概率（本模块不使用）
        actor_net_arch: Actor 网络结构
        critic_net_arch: Critic 网络结构
        share_encoder: 是否共享 Actor/Critic 的 encoder（本模块不使用）

    Returns:
        model: 训练后的模型
        trained_steps: 本次训练的步数
        total_trained_steps: 累计训练的步数
    """
    if actor_net_arch is None:
        actor_net_arch = [128, 128]
    if critic_net_arch is None:
        critic_net_arch = [128, 128]
    
    experiment_dir, checkpoints_dir, tensorboard_dir, save_dir = get_experiment_dirs(experiment_name)

    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # 与 attention1 保持一致的 hparams 格式，便于统一处理
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
        "features_dim": features_dim,
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "ffn_dim": ffn_dim,
        "dropout": dropout,
        "net_arch": dict(pi=actor_net_arch, vf=critic_net_arch)
    }

    policy_kwargs = dict(
        features_extractor_class=SimpleBundleExtractor,
        features_extractor_kwargs=dict(features_dim=hparams["features_dim"]),
        net_arch=hparams["net_arch"]
    )

    existing_steps = 0

    if model is None and resume and not overwrite:
        model, existing_steps = load_latest_checkpoint(experiment_name, env, policy_kwargs)

    # 使用可变对象存储 clip_range 值，以便在训练过程中可以修改
    current_clip_range = [clip_range]
    
    # clip_range 需要是可调用对象（函数）
    def clip_range_fn(_):
        return current_clip_range[0]
    
    if model is None:
        print(f"创建新模型，实验: {experiment_name}")
        print(f"超参数: learning_rate={learning_rate}, ent_coef={ent_coef}, clip_range={clip_range}")
        print(f"         n_steps={n_steps}, batch_size={batch_size}, gamma={gamma}")
        print(f"         target_kl={target_kl}")
        
        # 构建 PPO 参数
        ppo_kwargs = dict(
            policy=MultiInputActorCriticPolicy,
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
            tensorboard_log=tensorboard_dir
        )
        
        if target_kl is not None and isinstance(target_kl, (int, float)):
            ppo_kwargs['target_kl'] = target_kl
            print(f"设置 target_kl={target_kl}")
        
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
                self.current_clip_range = current_clip_range
                self.initial_clip_range = initial_clip_range
                self.final_clip_range = final_clip_range
                self.remaining_timesteps = remaining_timesteps
                self.total_timesteps = total_timesteps
                self.start_timesteps = total_timesteps - remaining_timesteps
            
            def _on_step(self):
                current_progress = self.num_timesteps - self.start_timesteps
                progress = min(current_progress / self.remaining_timesteps, 1.0) if self.remaining_timesteps > 0 else 1.0
                self.current_clip_range[0] = self.initial_clip_range - (self.initial_clip_range - self.final_clip_range) * progress
                return True
        
        clip_decay_callback = ClipRangeDecayCallback(current_clip_range, clip_range, remaining_timesteps, total_timesteps)
        callbacks.append(clip_decay_callback)
        print(f"启用 clip_range 衰减: 从 {clip_range} 线性衰减到 0.05")

    print(f"开始训练，剩余步数: {remaining_timesteps}/{total_timesteps}")
    if logger:
        logger.info(f"开始训练，剩余步数: {remaining_timesteps}/{total_timesteps}")

    print(f"model.policy: {model.policy}")

    model.learn(
        total_timesteps=remaining_timesteps,
        reset_num_timesteps=False,
        callback=callbacks,
        tb_log_name="log"
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