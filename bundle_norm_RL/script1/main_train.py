"""
Incremental Level Bundle RL 训练入口

使用 SB3 原生 MultiInputPolicy + 自定义 IncrementalLevelBundleFeaturesExtractor。
与 script/main_train.py 保持相同的风格。
"""

import os
from pathlib import Path

import yaml

from bundle_norm_RL.script1.config import LevelBundleConfig
from bundle_norm_RL.script1.logger import get_logger
from bundle_norm_RL.script1.env import IncrementalLevelBundleEnv
from bundle_norm_RL.script1.train import train


def load_train_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train_interleaved(
    logger, configs, rounds=3, steps_per_config_per_round=20_000,
    experiment_name="incremental_level_bundle_exp", K=20,
    learning_rate=1e-5, clip_range=0.1, clip_range_decay=True,
    n_steps=2048, batch_size=512, gamma=0.99, gae_lambda=0.95,
    n_epochs=3, ent_coef=0.005, vf_coef=1.0, max_grad_norm=0.5,
    target_kl=0.015, hidden_dim=128, log_std_init=-3.0, overwrite=False,
    encoder_type="deepset", n_heads=4, n_attn_layers=2,
):
    """交错训练函数：在多个 config 之间交替训练"""
    model = None

    for round_idx in range(rounds):
        logger.info(f"===== 训练轮次 {round_idx + 1}/{rounds} =====")

        for config_idx, config in enumerate(configs):
            logger.info(f"  训练 Config {config_idx + 1}/{len(configs)} (realization {config.n})")

            env = IncrementalLevelBundleEnv.create_env(logger, config, K=K)

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
                hidden_dim=hidden_dim,
                log_std_init=log_std_init,
                overwrite=overwrite,
                encoder_type=encoder_type,
                n_heads=n_heads,
                n_attn_layers=n_attn_layers,
            )

            env.close()

    return model


def create_config_list(config_dir: Path, i=2, t=1):
    """从指定目录加载配置文件"""
    configs = []
    for n in range(1):
        config_path = config_dir / f"config_{i}_{t}_{n}.pkl"
        if config_path.exists():
            configs.append(LevelBundleConfig.from_pkl(config_path))
        else:
            print(f"警告：配置文件不存在: {config_path}")
    return configs


def main(experiment_name, config_path=None):
    project_root = Path(__file__).parent.parent.absolute()  # bundle_norm_RL

    if config_path is None:
        config_path = Path(__file__).parent.absolute() / "train_config.yml"
    else:
        config_path = Path(config_path)

    config = load_train_config(config_path)

    log_dir = project_root / "train_result" / "model" / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    config_save_path = log_dir / f"{experiment_name}.yml"
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, encoding='utf-8')

    logger = get_logger(str(log_dir / "incremental_level_bundle_train.log"))
    logger.info(f"项目根目录: {project_root}")
    logger.info(f"配置文件: {config_path}")
    logger.info(f"完整配置: {config}")

    exp_config = config['experiment']
    env_config = config['environment']
    ppo_config = config['ppo']
    net_config = config['network']
    train_cfg = config['training']

    config_dir = project_root / "configs"
    train_configs = create_config_list(config_dir, i=2, t=2)
    logger.info(f"加载了 {len(train_configs)} 个配置")

    if not train_configs:
        logger.error("没有找到任何配置文件！")
        return

    model = train_interleaved(
        logger=logger,
        configs=train_configs,
        rounds=exp_config['rounds'],
        steps_per_config_per_round=train_cfg['steps_per_config_per_round'],
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
        hidden_dim=net_config['hidden_dim'],
        log_std_init=ppo_config.get('log_std_init', -3.0),
        overwrite=exp_config['overwrite'],
        encoder_type=net_config.get('encoder_type', 'deepset'),
        n_heads=net_config.get('n_heads', 4),
        n_attn_layers=net_config.get('n_attn_layers', 2),
    )

    logger.info("训练完成！")
    return model


if __name__ == "__main__":
    experiment_name = "inc_exp_01"
    main(experiment_name=experiment_name)
