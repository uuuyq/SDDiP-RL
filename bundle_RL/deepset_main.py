"""
DeepSet PPO 主训练入口

基于 dirichlet_main.py 的结构，使用 DeepSet 架构进行训练。

===========================================
              使用方法
===========================================

python deepset_main.py

===========================================
              DeepSet 特有参数
===========================================

- policy_type: 策略类型 ("deepset", "shared", "separate")
"""
import os
from pathlib import Path

import yaml

from bundle_RL.config import BundleConfig
from bundle_RL.script.logger import get_logger
from bundle_RL.script.deepset.env import BundleDualEnv
from bundle_RL.script.deepset.train import train


def load_train_config(config_path: str) -> dict:
    """加载训练配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train_interleaved(
    logger,
    configs,
    rounds=3,
    steps_per_config_per_round=20_000,
    experiment_name="deepset_exp",
    K=20,
    learning_rate=3e-4,
    clip_range=0.2,
    clip_range_decay=True,
    n_steps=512,
    batch_size=128,
    gamma=0.99,
    gae_lambda=0.95,
    n_epochs=10,
    ent_coef=0.005,
    vf_coef=0.5,
    max_grad_norm=0.5,
    target_kl=None,
    features_dim=128,
    hidden_dim=64,
    dropout=0.1,
    actor_net_arch=None,
    critic_net_arch=None,
    policy_type="deepset",
    overwrite=False
):
    """
    DeepSet 版本的交错训练函数

    Args:
        configs: 配置列表（每个 config 包含自己的 n 参数）
        rounds: 训练轮数
        steps_per_config_per_round: 每轮每个 config 训练的步数
        experiment_name: 实验名称
        K: 样本数量参数
        learning_rate: 学习率
        clip_range: PPO clip 范围
        clip_range_decay: 是否启用 clip_range 线性衰减
        n_steps: 每次更新采集的步数
        batch_size: 批大小
        gamma: 折扣因子
        gae_lambda: GAE 参数
        n_epochs: 训练轮数
        ent_coef: 熵系数
        vf_coef: 价值函数系数
        max_grad_norm: 最大梯度范数
        target_kl: KL 散度目标
        features_dim: 特征提取器维度
        hidden_dim: 编码器隐藏层维度（同时用于 Query/Key 维度）
        dropout: Dropout 概率
        actor_net_arch: Actor 网络结构
        critic_net_arch: Critic 网络结构
        policy_type: 策略类型 ("deepset", "shared", "separate")
        overwrite: 是否覆盖已有模型重新训练

    Returns:
        训练好的模型
    """
    if actor_net_arch is None:
        actor_net_arch = [128, 128]
    if critic_net_arch is None:
        critic_net_arch = [128, 128]

    model = None

    for round_idx in range(rounds):
        logger.info(f"===== 训练轮次 {round_idx + 1}/{rounds} =====")

        for config_idx, config in enumerate(configs):
            logger.info(f"  训练 Config {config_idx + 1}/{len(configs)} (realization {config.n})")

            # 创建当前 config 的环境
            env, _ = BundleDualEnv.create_env(logger, config, K=K)

            # 训练
            model, _, _ = train(
                env=env,
                logger=logger,
                model=model,
                total_timesteps=steps_per_config_per_round,
                experiment_name=experiment_name,
                ent_coef=ent_coef,
                learning_rate=learning_rate,
                clip_range=clip_range,
                clip_range_decay=clip_range_decay,
                n_steps=n_steps,
                batch_size=batch_size,
                gamma=gamma,
                gae_lambda=gae_lambda,
                n_epochs=n_epochs,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                target_kl=target_kl,
                features_dim=features_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                actor_net_arch=actor_net_arch,
                critic_net_arch=critic_net_arch,
                policy_type=policy_type,
                overwrite=overwrite
            )

            # 清理环境资源
            env.close()

    return model


def create_config_list(config_dir: Path, i=1, t=5):
    """
    从指定目录加载配置文件

    Args:
        config_dir: 配置文件目录
        i: 配置参数 i
        t: 配置参数 t

    Returns:
        配置对象列表
    """
    configs = []
    for n in range(6):
        config_path = config_dir / f"config_{i}_{t}_{n}.pkl"
        if config_path.exists():
            configs.append(BundleConfig.from_pkl(config_path))
        else:
            print(f"警告：配置文件不存在: {config_path}")
    return configs


def main(experiment_name, config_path=None, **kwargs):
    """
    主函数

    Args:
        experiment_name: 实验名称（必须指定）
        config_path: 配置文件路径（可选，默认为 deepset_config.yml）
        **kwargs: 其他训练参数
    """
    # 获取项目根目录的绝对路径
    project_root = Path(__file__).parent.absolute()

    # 加载配置文件（默认使用 DeepSet 配置）
    if config_path is None:
        config_path = project_root / "deepset_config.yml"
    else:
        config_path = Path(config_path)

    config = load_train_config(config_path)

    # 日志目录
    log_dir = project_root / "train_result" / "model" / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(str(log_dir / "deepset_bundle_env_train.log"))
    logger.info(f"项目根目录: {project_root}")
    logger.info(f"配置文件: {config_path}")
    logger.info(f"完整配置: {config}")

    # 提取配置参数
    exp_config = config['experiment']
    env_config = config['environment']
    ppo_config = config['ppo']
    net_config = config['network']
    train_config = config['training']

    # 创建 config 列表
    config_dir = project_root / "configs"
    train_configs = create_config_list(config_dir)
    logger.info(f"加载了 {len(train_configs)} 个配置")

    if not train_configs:
        logger.error("没有找到任何配置文件！")
        return

    # 交错训练
    model = train_interleaved(
        logger=logger,
        configs=train_configs,
        rounds=exp_config['rounds'],
        steps_per_config_per_round=train_config['steps_per_config_per_round'],
        experiment_name=experiment_name,
        K=env_config['K'],
        learning_rate=ppo_config['learning_rate'],
        clip_range=ppo_config['clip_range'],
        clip_range_decay=ppo_config['clip_range_decay'],
        n_steps=ppo_config['n_steps'],
        batch_size=ppo_config['batch_size'],
        gamma=ppo_config['gamma'],
        gae_lambda=ppo_config['gae_lambda'],
        n_epochs=ppo_config['n_epochs'],
        ent_coef=ppo_config['ent_coef'],
        vf_coef=ppo_config['vf_coef'],
        max_grad_norm=ppo_config['max_grad_norm'],
        target_kl=ppo_config.get('target_kl', None),
        features_dim=net_config['features_dim'],
        hidden_dim=net_config['hidden_dim'],
        dropout=net_config['dropout'],
        actor_net_arch=net_config['actor_net_arch'],
        critic_net_arch=net_config['critic_net_arch'],
        policy_type="deepset",  # 使用新的 DeepSetActorCriticPolicy
        overwrite=exp_config['overwrite']
    )

    logger.info("DeepSet PPO 训练完成！")
    return model


if __name__ == "__main__":
    # ========================================================
    # 训练参数配置（在这里调整实验名称和训练参数）
    # ========================================================
    experiment_name = "exp_deepset_01"          # 实验名称
    config_path = None                            # 配置文件路径（None 表示使用默认配置 deepset_config.yml）

    # ========================================================
    # 启动训练
    # ========================================================
    main(
        experiment_name=experiment_name,
        config_path=config_path
    )