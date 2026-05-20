import os
from pathlib import Path

from bundle_RL.config import BundleConfig
from bundle_RL.script.logger import get_logger
from bundle_RL.script.utils import create_env
from bundle_RL.script.mask.train import train

def train_interleaved(logger, configs, rounds=3, steps_per_config_per_round=20_000, experiment_name="multi_config_exp"):
    """
    交错训练函数：在多个 config 之间交替训练

    Args:
        logger: 日志器
        configs: 配置列表（每个 config 包含自己的 n 参数）
        rounds: 训练轮数（每个 config 会被训练 rounds 次）
        steps_per_config_per_round: 每轮每个 config 训练的步数
        experiment_name: 实验名称，用于区分不同实验

    Returns:
        训练好的模型
    """
    model = None

    for round_idx in range(rounds):
        logger.info(f"===== 训练轮次 {round_idx + 1}/{rounds} =====")

        for config_idx, config in enumerate(configs):
            logger.info(f"  训练 Config {config_idx + 1}/{len(configs)} (realization {config.n})")

            # 创建当前 config 的环境（n 已包含在 config 中）
            env, _ = create_env(logger, config)

            # 训练（如果 model 已存在则继续训练）
            # train() 返回 (model, remaining_timesteps, total_trained_steps)，只取模型
            model, _, _ = train(
                env=env,
                logger=logger,
                model=model,
                total_timesteps=steps_per_config_per_round,
                experiment_name=experiment_name,
                ent_coef=0.01  # 尝试微量探索
            )

    return model


def main(experiment_name):
    log_dir = os.path.join("train_result", experiment_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = get_logger(os.path.join(log_dir, "bundle_env_train.log"))
    
    # ===============================
    # 2️⃣ 创建 config 列表
    # ===============================
    train_configs = create_config_list()
    logger.info(f"加载了 {len(train_configs)} 个配置")
    
    # ===============================
    # 3️⃣ 交错训练
    # ===============================
    # 训练参数：3 轮 × 3 个 config × 每 config 20,000 步 = 180,000 总步数
    model = train_interleaved(
        logger=logger,
        configs=train_configs,
        rounds=3,
        steps_per_config_per_round=20_000,
        experiment_name=experiment_name
    )
    


def create_config_list():
    configs = []
    i = 1
    for t in range(1, 24):
        for n in range(6):
            config_path = Path(f"./configs/config_{i}_{t}_{n}.pkl")
            configs.append(BundleConfig.from_pkl(config_path))

    return configs




if __name__ == "__main__":
    experiment_name = "multi_config_exp_03"  # 实验名称，用于区分不同实验
    main(experiment_name)
    



"""
--------------------------------------------------
[Rollout 阶段 - 业务表现]
- ep_rew_mean: 
    含义: 回合平均总奖励。
    判断: 核心指标，必须长期看涨。如果不涨，检查 Reward 函数。
- ep_len_mean: 
    含义: 回合平均长度。
    判断: 判定模型是“早死”还是“通关”。

[Train 阶段 - 模型稳定性]
- entropy_loss: 
    含义: 策略熵（动作随机性）。
    判断: 绝对值应缓慢下降。绝对值过快趋近0表示过早收敛（不再尝试新动作）；
         一直很大表示模型在乱撞，学不到规律。
- explained_variance: 
    含义: 预测奖励的解释方差。
    判断: 越接近 1.0 越好。如果小于 0，说明 Critic 网络预测得比瞎猜还差。
- approx_kl: 
    含义: 新旧策略的 KL 散度（策略更新步长）。
    判断: 理想在 0.001 到 0.05 之间。若过大（如 >0.1），训练易崩溃。
- clip_fraction: 
    含义: 触发 PPO 截断机制的比例。
    判断: 常用 0.1~0.2。如果过高，说明更新被频繁强制限制。
- value_loss: 
    含义: 价值函数误差。
    判断: 代表评价员准不准，通常先升后降。

[Time 阶段 - 性能]
- fps: 
    含义: 每秒处理步数。
    判断: 衡量环境执行速度，主要受 env.step() 的复杂度影响。
"""