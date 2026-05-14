from pathlib import Path

from matplotlib import pyplot as plt

from bundle_RL.bundle_env import BundleDualEnv
from bundle_RL.config import BundleConfig
from bundle_RL.lag_problem import MasterProblem
from bundle_RL.logger import get_logger
from train import train


def create_config_list():
    """创建多个 config 列表，用于交错训练"""
    # ===============================
    # Config 1 (原始配置，realization 0)
    # ===============================
    config1 = BundleConfig(
        T=5,
        N_VARS=13,
        X_TRIAL=[-0.0, 1.0, 1.0],
        Y_TRIAL=[0.0, 131.60809087723158, 45.0],
        X_BS_TRIAL=[[-0.0, 0.0], [1.0, 1.0], [1.0, 1.0]],
        SOC_TRIAL=[0.0],
        PATH=Path(r"..\data\01_test_cases\case6ww\t24_n06"),
        n=0,  # realization 索引
    )
    config2 = BundleConfig(
        T=5,
        N_VARS=13,
        X_TRIAL=[-0.0, 1.0, 1.0],
        Y_TRIAL=[0.0, 131.60809087723158, 45.0],
        X_BS_TRIAL=[[-0.0, 0.0], [1.0, 1.0], [1.0, 1.0]],
        SOC_TRIAL=[0.0],
        PATH=Path(r"..\data\01_test_cases\case6ww\t24_n06"),
        n=1,  # realization 索引
    )
    config3 = BundleConfig(
        T=5,
        N_VARS=13,
        X_TRIAL=[-0.0, 1.0, 1.0],
        Y_TRIAL=[0.0, 131.60809087723158, 45.0],
        X_BS_TRIAL=[[-0.0, 0.0], [1.0, 1.0], [1.0, 1.0]],
        SOC_TRIAL=[0.0],
        PATH=Path(r"..\data\01_test_cases\case6ww\t24_n06"),
        n=2,  # realization 索引
    )
    config4 = BundleConfig(
        T=5,
        N_VARS=13,
        X_TRIAL=[-0.0, 1.0, 1.0],
        Y_TRIAL=[0.0, 131.60809087723158, 45.0],
        X_BS_TRIAL=[[-0.0, 0.0], [1.0, 1.0], [1.0, 1.0]],
        SOC_TRIAL=[0.0],
        PATH=Path(r"..\data\01_test_cases\case6ww\t24_n06"),
        n=3,  # realization 索引
    )
    config5 = BundleConfig(
        T=5,
        N_VARS=13,
        X_TRIAL=[-0.0, 1.0, 1.0],
        Y_TRIAL=[0.0, 131.60809087723158, 45.0],
        X_BS_TRIAL=[[-0.0, 0.0], [1.0, 1.0], [1.0, 1.0]],
        SOC_TRIAL=[0.0],
        PATH=Path(r"..\data\01_test_cases\case6ww\t24_n06"),
        n=4,  # realization 索引
    )
    config6 = BundleConfig(
        T=5,
        N_VARS=13,
        X_TRIAL=[-0.0, 1.0, 1.0],
        Y_TRIAL=[0.0, 131.60809087723158, 45.0],
        X_BS_TRIAL=[[-0.0, 0.0], [1.0, 1.0], [1.0, 1.0]],
        SOC_TRIAL=[0.0],
        PATH=Path(r"..\data\01_test_cases\case6ww\t24_n06"),
        n=5,  # realization 索引
    )



    return [config1, config2, config3, config4, config5], [config6]


def create_env(logger, config):
    """创建单个环境（使用 config 中的 n 参数）"""
    state_dim = config.N_VARS
    K = 20

    env = BundleDualEnv(
        logger=logger,
        config=config,
        n=config.n,  # 直接使用 config 中的 realization 索引
        state_dim=state_dim,
        K=K
    )
    master = MasterProblem(logger, config.N_VARS, tolerance=1e-5)
    
    return env, master


def test(env, model, master, logger):
    delta_history = []
    reward_history = []

    obs, _ = env.reset()

    sub_result = env.bundle[-1]
    x_new = sub_result["pi"]
    f_new = sub_result["phi"]
    g_new = sub_result["g"]
    # 第一次，更新 f_best和x_best
    master.update_strategy(x_new, f_new, g_new, ub=None)

    logger.info("==== ROLLOUT ====")
    for step in range(20):
        master.add_cut(x_new, f_new, g_new)
        ub, _ = master.solve_master()
        _, delta, stop_flag = master.update_strategy(x_new, f_new, g_new, ub=ub)
        action, _ = model.predict(obs, deterministic=True)
        state, reward, terminated, truncated, info = env.step(action)
        # 保存数据
        logger.info(f"delta: {delta}")
        delta_history.append(delta)
        reward_history.append(reward)
        # 获取新的子问题得到的cut
        sub_result = env.bundle[-1]
        x_new = sub_result["pi"]
        f_new = sub_result["phi"]
        g_new = sub_result["g"]
        logger.info(f"reward: {reward}")

    logger.info("Test finished successfully.")

    # 绘制曲线
    plt.figure(figsize=(12, 5))
    # 绘制 Delta 收敛图
    plt.subplot(1, 2, 1)
    plt.plot(delta_history, marker='o', color='b', label='Delta (Gap)')
    plt.yscale('log')  # 通常 delta 跨度很大，建议开启对数坐标
    plt.xlabel('Iteration Step')
    plt.ylabel('Delta Value (Log Scale)')
    plt.title('Convergence of Bundle Method')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    # 绘制 Reward 变化图
    plt.subplot(1, 2, 2)
    plt.plot(reward_history, marker='s', color='r', label='Step Reward')
    plt.xlabel('Iteration Step')
    plt.ylabel('Reward')
    plt.title('Reward during Rollout')
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


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
            model = train(
                env=env,
                logger=logger,
                model=model,
                total_timesteps=steps_per_config_per_round,
                experiment_name=experiment_name
            )
    
    return model


def main():
    logger = get_logger("log/bundle_env_test.log")
    
    # ===============================
    # 1️⃣ 实验配置
    # ===============================
    experiment_name = "multi_config_exp_01"  # 实验名称，用于区分不同实验
    
    # ===============================
    # 2️⃣ 创建 config 列表
    # ===============================
    train_configs, test_configs = create_config_list()
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
    
    # ===============================
    # 4️⃣ 在第一个 config 上测试
    # ===============================
    test_env, test_master = create_env(logger, test_configs[0])
    test(test_env, model, test_master, logger)


if __name__ == "__main__":
    main()



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