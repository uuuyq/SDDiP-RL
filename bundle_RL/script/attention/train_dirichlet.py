"""
Dirichlet PPO 训练脚本

基于 stable-baselines3 的 PPO，使用 Dirichlet 分布作为策略输出。

主要改进：
1. 使用 Dirichlet 分布替代 softmax 输出 lambda 权重
2. Attention encoder 输出后添加 mask
3. 自定义策略网络

===========================================
              依赖
===========================================

需要同时使用：
- encoder_v2.py: 带输出 mask 的 encoder
- dirichlet_policy.py: Dirichlet 策略头
"""
import os
import re
import csv
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

from bundle_RL.script.attention.encoder_v2 import AttentionBundleEncoder
from bundle_RL.script.attention.dirichlet_policy import (
    DirichletPolicyHead,
    DirichletCombinedDistribution
)


# ============================
# Dirichlet Features Extractor
# ============================

class DirichletFeaturesExtractor(BaseFeaturesExtractor):
    """
    Dirichlet 版本的 Features Extractor

    使用 encoder_v2 中的 AttentionBundleEncoder，
    并在输出时确保无效位置为零向量。
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 128,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        ffn_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__(observation_space, features_dim)

        # 从 observation_space 获取维度信息
        state_dim = observation_space["pi"].shape[0]
        trial_point_dim = observation_space["trial_point"].shape[0]
        realization_dim = observation_space["realization"].shape[0]
        K = observation_space["cuts"].shape[0]

        self.encoder = AttentionBundleEncoder(
            state_dim=state_dim,
            trial_point_dim=trial_point_dim,
            realization_dim=realization_dim,
            K=K,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            dropout=dropout
        )

        self.K = K
        self.hidden_dim = hidden_dim

    def forward(self, observations: dict) -> torch.Tensor:
        """
        前向传播

        Args:
            observations: dict，包含 cuts, valid_mask, pi, trial_point, realization

        Returns:
            features: (batch_size, features_dim) 全局特征向量
        """
        cuts = observations["cuts"]
        valid_mask = observations["valid_mask"]
        pi = observations["pi"]
        trial_point = observations["trial_point"]
        realization = observations["realization"]

        # 使用 encoder_v2（已包含输出 mask）
        cut_embeddings, global_embedding, cls_embedding = self.encoder(
            cuts=cuts,
            valid_mask=valid_mask,
            pi=pi,
            trial_point=trial_point,
            realization=realization
        )

        # 返回 cls_embedding 作为全局特征
        return cls_embedding


class DirichletExtractorWithCuts(BaseFeaturesExtractor):
    """
    Dirichlet 版本的 Features Extractor（包含 cut embeddings）

    除了返回全局特征，还返回 cut embeddings 用于 Dirichlet 策略头。
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 128,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        ffn_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__(observation_space, features_dim)

        state_dim = observation_space["pi"].shape[0]
        trial_point_dim = observation_space["trial_point"].shape[0]
        realization_dim = observation_space["realization"].shape[0]
        K = observation_space["cuts"].shape[0]

        self.encoder = AttentionBundleEncoder(
            state_dim=state_dim,
            trial_point_dim=trial_point_dim,
            realization_dim=realization_dim,
            K=K,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            dropout=dropout
        )

        self.K = K
        self.hidden_dim = hidden_dim

        # 投影层：将 hidden_dim 投影到 features_dim
        self.projection = nn.Linear(hidden_dim, features_dim)

        # 存储中间结果供策略头使用
        self._last_cut_embeddings = None
        self._last_valid_mask = None

    def forward(self, observations: dict) -> torch.Tensor:
        """
        前向传播

        Returns:
            features: (batch_size, features_dim)
        """
        cuts = observations["cuts"]
        valid_mask = observations["valid_mask"]
        pi = observations["pi"]
        trial_point = observations["trial_point"]
        realization = observations["realization"]

        cut_embeddings, global_embedding, cls_embedding = self.encoder(
            cuts=cuts,
            valid_mask=valid_mask,
            pi=pi,
            trial_point=trial_point,
            realization=realization
        )

        # 存储中间结果（不 detach，保留梯度）
        self._last_cut_embeddings = cut_embeddings
        self._last_valid_mask = valid_mask

        # 将 cls_embedding 投影到 features_dim
        return self.projection(cls_embedding)


# ============================
# Dirichlet Actor-Critic Policy
# ============================

class DirichletActorCriticPolicy(ActorCriticPolicy):
    """
    Dirichlet 版本的 Actor-Critic 策略

    使用自定义的 Dirichlet 策略头输出 lambda 权重。
    """

    def __init__(
        self,
        *args,
        K: int = 20,
        dirichlet_hidden_dim: int = 64,
        min_alpha: float = 1.0,
        eta_scale: float = 1.0,
        **kwargs
    ):
        # 保存 Dirichlet 参数
        self.K = K
        self.dirichlet_hidden_dim = dirichlet_hidden_dim
        self.min_alpha = min_alpha
        self.eta_scale = eta_scale

        # 调用父类 __init__（这会初始化 mlp_extractor）
        super().__init__(*args, **kwargs)

        # Dirichlet 策略头（使用 cls_embedding）
        # 从 net_arch 获取 latent 维度
        input_dim = None
        if hasattr(self, 'net_arch') and 'pi' in self.net_arch:
            # 使用 actor 网络架构的最后一层维度
            actor_arch = self.net_arch['pi']
            if actor_arch and len(actor_arch) > 0:
                input_dim = actor_arch[-1]
        
        if input_dim is None:
            # 备选方案：使用 features_dim
            input_dim = self.features_extractor.features_dim

        # 确保 input_dim 是有效的整数
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError(f"Invalid input_dim: {input_dim} (type: {type(input_dim)}). net_arch: {getattr(self, 'net_arch', None)}")

        self.dirichlet_head = DirichletPolicyHead(
            input_dim=input_dim,
            K=K,
            hidden_dim=dirichlet_hidden_dim,
            min_alpha=min_alpha,
            eta_scale=eta_scale
        )

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> DirichletCombinedDistribution:
        """
        从 latent 变量获取 Dirichlet 分布

        Args:
            latent_pi: (batch_size, latent_dim)

        Returns:
            DirichletCombinedDistribution 对象
        """
        # 从 observations 获取 valid_mask
        if hasattr(self, '_last_observations') and self._last_observations is not None:
            obs = self._last_observations
            if isinstance(obs, dict) and 'valid_mask' in obs:
                valid_mask = obs['valid_mask']
            else:
                valid_mask = torch.ones(latent_pi.shape[0], self.K, device=latent_pi.device)
        else:
            valid_mask = torch.ones(latent_pi.shape[0], self.K, device=latent_pi.device)

        # 将 valid_mask 转换为 float 类型（与 latent_pi 类型一致）
        valid_mask = valid_mask.to(latent_pi.dtype)

        concentration, eta, _ = self.dirichlet_head(latent_pi, valid_mask)

        return DirichletCombinedDistribution(
            concentration=concentration,
            eta=eta,
            valid_mask=valid_mask,
            lambda_temp=1.0
        )

    def forward(self, obs, deterministic=False):
        """
        前向传播

        Args:
            obs: observations
            deterministic: 是否使用确定性策略

        Returns:
            actions, values, log_probs
        """
        # 提取特征
        features = self.extract_features(obs)
        self._last_observations = obs

        # 获取 latent 变量
        latent_pi = self.mlp_extractor.forward_actor(features)
        latent_vf = self.mlp_extractor.forward_critic(features)

        # 获取分布
        distribution = self._get_action_dist_from_latent(latent_pi)

        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.sample()

        log_probs = distribution.log_prob(actions)
        values = self.value_net(latent_vf)

        return actions, values, log_probs

    def evaluate_actions(self, obs, actions):
        """
        评估动作

        Args:
            obs: observations
            actions: 要评估的动作

        Returns:
            values, log_probs, entropy
        """
        features = self.extract_features(obs)
        self._last_observations = obs

        latent_pi = self.mlp_extractor.forward_actor(features)
        latent_vf = self.mlp_extractor.forward_critic(features)

        distribution = self._get_action_dist_from_latent(latent_pi)

        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.value_net(latent_vf)

        return values, log_probs, entropy


# ============================
# 训练函数
# ============================

def get_experiment_dirs(experiment_name: str):
    """获取实验相关的目录路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(current_dir))

    experiment_dir = os.path.join(base_dir, "train_result", "model", experiment_name)
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    tensorboard_dir = os.path.join(base_dir, "train_result", "ppo_tensorboard", experiment_name)
    save_dir = os.path.join(experiment_dir, "save")

    return experiment_dir, checkpoints_dir, tensorboard_dir, save_dir


def extract_steps_from_checkpoint(filename: str) -> int:
    """从 checkpoint 文件名中提取步数"""
    match = re.search(r'_(\d+)_steps', filename)
    if match:
        return int(match.group(1))
    return 0


def train(
    env,
    save_path=None,
    logger=None,
    model=None,
    total_timesteps=200_000,
    checkpoint_freq=5000,
    experiment_name="dirichlet_default",
    ent_coef=0.005,
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
    features_dim=128,
    hidden_dim=64,
    num_heads=4,
    num_layers=1,
    ffn_dim=128,
    dropout=0.1,
    K=20,
    dirichlet_hidden_dim=64,
    min_alpha=1.0,
    eta_scale=1.0,
    actor_net_arch=None,
    critic_net_arch=None
):
    """
    Dirichlet PPO 训练函数

    Args:
        env: 环境
        save_path: 模型保存路径（可选）
        logger: 日志器
        model: 已有模型（用于续训）
        total_timesteps: 总训练步数
        checkpoint_freq: 检查点保存频率
        experiment_name: 实验名称
        resume: 是否续训
        overwrite: 是否覆盖已有模型
        其他参数同 standard PPO 训练

    Dirichlet 特有参数:
        K: cut 数量
        dirichlet_hidden_dim: Dirichlet 策略头隐藏层维度
        min_alpha: Dirichlet concentration 最小值
        eta_scale: 步长缩放因子

    Returns:
        model: 训练好的模型
        remaining_timesteps: 本次训练的步数
        total_trained_steps: 累计训练步数
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

    print(f"=" * 60)
    print(f"Dirichlet PPO 训练")
    print(f"实验名称: {experiment_name}")
    print(f"=" * 60)

    # 超参数字典（用于记录）
    hparams = {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "n_epochs": n_epochs,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "target_kl": target_kl,
        "features_dim": features_dim,
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "ffn_dim": ffn_dim,
        "dropout": dropout,
        "K": K,
        "dirichlet_hidden_dim": dirichlet_hidden_dim,
        "min_alpha": min_alpha,
        "eta_scale": eta_scale,
        "actor_net_arch": str(actor_net_arch),
        "critic_net_arch": str(critic_net_arch)
    }

    # 检查是否需要续训
    existing_steps = 0
    if model is None and resume and not overwrite:
        # 查找最近的检查点
        import glob
        checkpoints = glob.glob(os.path.join(checkpoints_dir, "dirichlet_ppo_checkpoint_*.zip"))
        if checkpoints:
            # 按步数排序
            checkpoints.sort(key=extract_steps_from_checkpoint, reverse=True)
            latest_checkpoint = checkpoints[0]
            existing_steps = extract_steps_from_checkpoint(latest_checkpoint)
            print(f"找到检查点: {latest_checkpoint}, 已训练步数: {existing_steps}")
            # 从检查点加载模型
            model = PPO.load(latest_checkpoint, env=env)
            print(f"已从检查点恢复模型")

    # 构建策略参数
    features_extractor_kwargs = dict(
        features_dim=features_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_dim=ffn_dim,
        dropout=dropout
    )

    policy_kwargs = dict(
        features_extractor_class=DirichletExtractorWithCuts,
        features_extractor_kwargs=features_extractor_kwargs,
        net_arch=dict(pi=actor_net_arch, vf=critic_net_arch),
        K=K,
        dirichlet_hidden_dim=dirichlet_hidden_dim,
        min_alpha=min_alpha,
        eta_scale=eta_scale
    )

    # 创建模型
    if model is None:
        print(f"创建新 Dirichlet PPO 模型")
        print(f"Features: dim={features_dim}, hidden={hidden_dim}, heads={num_heads}, layers={num_layers}")
        print(f"Dirichlet: hidden={dirichlet_hidden_dim}, min_alpha={min_alpha}, eta_scale={eta_scale}")
        print(f"PPO: lr={learning_rate}, n_steps={n_steps}, batch={batch_size}, gamma={gamma}")

        # Clip range 函数（用于衰减）
        current_clip_range = [clip_range]

        def clip_range_fn(_):
            return current_clip_range[0]

        ppo_kwargs = dict(
            policy=DirichletActorCriticPolicy,
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

        # Clip range 函数（用于衰减）
        current_clip_range = [clip_range]

        def clip_range_fn(_):
            return current_clip_range[0]

        model.clip_range = clip_range_fn

    # 检查是否已完成训练
    if existing_steps >= total_timesteps:
        msg = f"模型已训练 {existing_steps} 步（目标 {total_timesteps} 步），无需继续训练"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        return model, 0, existing_steps

    remaining_timesteps = total_timesteps - existing_steps

    # 回调函数
    from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

    callbacks = [CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoints_dir,
        name_prefix="dirichlet_ppo_checkpoint"
    )]

    if clip_range_decay:
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

    # 训练
    print(f"开始训练，剩余步数: {remaining_timesteps}/{total_timesteps}")
    if logger:
        logger.info(f"开始训练，剩余步数: {remaining_timesteps}/{total_timesteps}")

    model.learn(
        total_timesteps=remaining_timesteps,
        reset_num_timesteps=False,
        callback=callbacks,
        tb_log_name="log"
    )

    total_trained_steps = existing_steps + remaining_timesteps

    # 保存模型
    timestamp = datetime.now().strftime("%m%d_%H%M")
    model_name = save_path if save_path else f"dirichlet_ppo_{timestamp}"
    final_save_path = os.path.join(save_dir, model_name)
    model.save(final_save_path)

    # 记录训练日志到 CSV
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
