"""
Level Bundle RL 训练入口

遵循 bundle_RL/script/attention2/main_train.py 的代码风格:
- YAML 配置加载
- 交错训练（多 config 交替）
- 配置文件保存
"""

import os
from pathlib import Path

import yaml

from bundle_norm_RL.script.config import LevelBundleConfig
from bundle_norm_RL.script.logger import get_logger
from bundle_norm_RL.script.env import LevelBundleEnv
from bundle_norm_RL.script.train import train


def load_train_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train_interleaved(
    logger, configs, rounds=3, steps_per_config_per_round=20_000,
    experiment_name="level_bundle_exp", K=20,
    learning_rate=3e-4, clip_range=0.2, clip_range_decay=True,
    n_steps=512, batch_size=128, gamma=0.99, gae_lambda=0.95,
    n_epochs=10, ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5,
    target_kl=None, hidden_dim=64, overwrite=False,
):
    """交错训练函数：在多个 config 之间交替训练"""
    model = None

    for round_idx in range(rounds):
        logger.info(f"===== 训练轮次 {round_idx + 1}/{rounds} =====")

        for config_idx, config in enumerate(configs):
            logger.info(f"  训练 Config {config_idx + 1}/{len(configs)} (realization {config.n})")

            env = LevelBundleEnv.create_env(logger, config, K=K)

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
                overwrite=overwrite,
            )

            env.close()

    return model


def create_config_list(config_dir: Path, i=1, t=5):
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

    logger = get_logger(str(log_dir / "level_bundle_train.log"))
    logger.info(f"项目根目录: {project_root}")
    logger.info(f"配置文件: {config_path}")
    logger.info(f"完整配置: {config}")

    exp_config = config['experiment']
    env_config = config['environment']
    ppo_config = config['ppo']
    net_config = config['network']
    train_cfg = config['training']

    config_dir = project_root / "configs"
    train_configs = create_config_list(config_dir, i=1, t=5)
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
        overwrite=exp_config['overwrite'],
    )

    logger.info("训练完成！")
    return model


if __name__ == "__main__":
    experiment_name = "exp_08"
    main(experiment_name=experiment_name)
