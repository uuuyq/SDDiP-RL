import os
from pathlib import Path

from bundle_RL.config import BundleConfig
from bundle_RL.script.logger import get_logger


def train_interleaved(logger, configs, rounds=3, steps_per_config_per_round=20_000, experiment_name="multi_config_exp", 
                      ent_coef=0, K=20, learning_rate=3e-4, clip_range=0.2, clip_range_decay=True):
    """
    交错训练函数：在多个 config 之间交替训练

    Args:
        logger: 日志器
        configs: 配置列表（每个 config 包含自己的 n 参数）
        rounds: 训练轮数（每个 config 会被训练 rounds 次）
        steps_per_config_per_round: 每轮每个 config 训练的步数
        experiment_name: 实验名称，用于区分不同实验
        ent_coef: 熵系数，控制探索程度
        K: 样本数量参数
        learning_rate: 学习率
        clip_range: PPO clip 范围
        clip_range_decay: 是否启用 clip_range 线性衰减

    Returns:
        训练好的模型
    """
    model = None

    for round_idx in range(rounds):
        logger.info(f"===== 训练轮次 {round_idx + 1}/{rounds} =====")

        for config_idx, config in enumerate(configs):
            logger.info(f"  训练 Config {config_idx + 1}/{len(configs)} (realization {config.n})")

            # 创建当前 config 的环境（n 已包含在 config 中）
            env, _ = BundleDualEnv.create_env(logger, config, K=K)

            # 训练（如果 model 已存在则继续训练）
            # train() 返回 (model, remaining_timesteps, total_trained_steps)，只取模型
            model, _, _ = train(
                env=env,
                logger=logger,
                model=model,
                total_timesteps=steps_per_config_per_round,
                experiment_name=experiment_name,
                ent_coef=ent_coef,
                learning_rate=learning_rate,
                clip_range=clip_range,
                clip_range_decay=clip_range_decay
            )

    return model


def main(experiment_name, ent_coef, K, steps_per_config_per_round, rounds, 
         learning_rate=3e-4, clip_range=0.2, clip_range_decay=True):
    # 获取项目根目录的绝对路径
    project_root = Path(__file__).parent.absolute()
    
    # 日志目录
    log_dir = project_root / "train_result" / "model" / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = get_logger(str(log_dir / "bundle_env_train.log"))
    logger.info(f"项目根目录: {project_root}")
    logger.info(f"训练参数: experiment_name={experiment_name}, ent_coef={ent_coef}, K={K}, "
                f"steps_per_config_per_round={steps_per_config_per_round}, rounds={rounds}, "
                f"learning_rate={learning_rate}, clip_range={clip_range}, clip_range_decay={clip_range_decay}")
    
    # ===============================
    # 创建 config 列表
    # ===============================
    config_dir = project_root / "configs"
    train_configs = create_config_list(config_dir)
    logger.info(f"加载了 {len(train_configs)} 个配置")
    
    if not train_configs:
        logger.error("没有找到任何配置文件！")
        return
    
    # ===============================
    # 交错训练
    # ===============================
    # 训练参数：3 轮 × 多个 config × 每 config 20,000 步
    model = train_interleaved(
        logger=logger,
        configs=train_configs,
        rounds=rounds,
        steps_per_config_per_round=steps_per_config_per_round,
        experiment_name=experiment_name,
        ent_coef=ent_coef,
        K=K,
        learning_rate=learning_rate,
        clip_range=clip_range,
        clip_range_decay=clip_range_decay
    )
    
    logger.info("训练完成！")
    return model


def create_config_list(config_dir):
    """从指定目录加载配置文件"""
    configs = []
    i = 1
    t=5
    for n in range(6):
        config_path = config_dir / f"config_{i}_{t}_{n}.pkl"
        if config_path.exists():
            configs.append(BundleConfig.from_pkl(config_path))
        else:
            print(f"警告：配置文件不存在: {config_path}")
    return configs

from bundle_RL.script.attention.train import train
from bundle_RL.script.attention.env import BundleDualEnv
if __name__ == "__main__":
    # ========================================================
    # 训练参数配置
    # 基于第三轮实验分析结果，选择最佳可信配置（exp21）
    # 要点：lr=5e-5稳定训练 + 低ent防止entropy上升 + 足够rounds充分收敛
    # ========================================================
    
    experiment_name = "exp21"          # 实验名称
    ent_coef = 0.005                   # 低熵系数，确保entropy下降、std收敛
    K = 10                             # 样本数量参数
    steps_per_config_per_round = 2000  # 每轮每个config训练步数（小步数更稳定）
    rounds = 30                        # 训练轮数（总步数 = 10×2000×30 = 600K）
    learning_rate = 5e-5               # 低学习率，避免KL散度过高
    clip_range = 0.2                   # PPO clip范围
    clip_range_decay = False           # 不启用clip衰减（exp12证明关闭更好）
    
    # ========================================================
    # 启动训练
    # ========================================================
    main(
        experiment_name=experiment_name,
        ent_coef=ent_coef,
        K=K,
        steps_per_config_per_round=steps_per_config_per_round,
        rounds=rounds,
        learning_rate=learning_rate,
        clip_range=clip_range,
        clip_range_decay=clip_range_decay
    )
