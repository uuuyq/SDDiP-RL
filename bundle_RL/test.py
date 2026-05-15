import json
import os
import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from bundle_RL.config import BundleConfig
from bundle_RL.logger import get_logger
from bundle_RL.utils import create_env
from bundle_RL.lag_problem import SubProblem, MasterProblem


def bundle_baseline(logger, config, tolerance=1e-5):
    """传统 Bundle 算法求解作为 baseline"""
    sub = SubProblem(logger, config, n=config.n)
    master = MasterProblem(logger, config.N_VARS, tolerance=tolerance)
    
    delta_history = []
    time_history = []
    x_new = np.zeros(config.N_VARS)
    g_new, f_new = sub.solve(x_new)
    master.update_strategy(x_new, f_new, g_new, ub=None)
    
    for i in range(20):
        start_time = time.time()
        master.add_cut(x_new, f_new, g_new)
        ub, x_new = master.solve_master()
        g_new, f_new = sub.solve(x_new)
        serious_step, delta, stop_flag = master.update_strategy(x_new, f_new, g_new, ub)
        end_time = time.time()
        
        delta_history.append(delta)
        time_history.append(end_time - start_time)
        logger.info(f"Baseline - delta: {delta}, time: {time_history[-1]:.4f}s")
        if stop_flag:
            break
    
    return delta_history, time_history


def bundle_RL(env, model, master, logger):
    delta_history = []
    reward_history = []
    time_history = []

    obs, _ = env.reset()

    sub_result = env.bundle[-1]
    x_new = sub_result["pi"]
    f_new = sub_result["phi"]
    g_new = sub_result["g"]
    # 第一次，更新 f_best和x_best
    master.update_strategy(x_new, f_new, g_new, ub=None)

    logger.info("==== ROLLOUT ====")
    for step in range(20):
        start_time = time.time()
        master.add_cut(x_new, f_new, g_new)
        ub, _ = master.solve_master()
        _, delta, stop_flag = master.update_strategy(x_new, f_new, g_new, ub=ub)
        action, _ = model.predict(obs, deterministic=True)
        state, reward, terminated, truncated, info = env.step(action)
        end_time = time.time()
        
        # 保存数据
        delta_history.append(delta)
        reward_history.append(reward)
        time_history.append(end_time - start_time)
        logger.info(f"delta: {delta}, reward: {reward}, time: {time_history[-1]:.4f}s")
        
        # 获取新的子问题得到的cut
        sub_result = env.bundle[-1]
        x_new = sub_result["pi"]
        f_new = sub_result["phi"]
        g_new = sub_result["g"]
        
        # 添加终止条件（与baseline保持一致）
        if stop_flag:
            logger.info(f"RL Model - 满足终止条件，提前停止，delta: {delta}")
            break

    logger.info("Test finished successfully.")
    
    return delta_history, reward_history, time_history


def bundle_RL_warmstart(env, model, master, logger, warmstart_threshold=1e-6, patience=3):
    """
    Warm-start 测试方法：当 RL 的 delta 不再发生变化时，切换成 baseline 的计算方式
    
    Args:
        env: 环境
        model: RL 模型
        master: MasterProblem 对象
        logger: 日志记录器
        warmstart_threshold: delta 变化阈值，小于此值认为不再变化
        patience: 连续多少次 delta 变化小于阈值后切换到 baseline
    
    Returns:
        delta_history: delta 历史记录
        reward_history: 奖励历史记录（仅RL阶段）
        time_history: 时间历史记录
        switch_step: 切换到 baseline 的步骤（None表示未切换）
    """
    delta_history = []
    reward_history = []
    time_history = []
    switch_step = None
    consecutive_small_changes = 0
    
    obs, _ = env.reset()

    sub_result = env.bundle[-1]
    x_new = sub_result["pi"]
    f_new = sub_result["phi"]
    g_new = sub_result["g"]
    master.update_strategy(x_new, f_new, g_new, ub=None)

    logger.info("==== WARMSTART ROLLOUT ====")
    
    # RL阶段
    for step in range(20):
        start_time = time.time()
        master.add_cut(x_new, f_new, g_new)
        ub, _ = master.solve_master()
        _, delta, stop_flag = master.update_strategy(x_new, f_new, g_new, ub=ub)
        action, _ = model.predict(obs, deterministic=True)
        state, reward, terminated, truncated, info = env.step(action)
        end_time = time.time()
        
        delta_history.append(delta)
        reward_history.append(reward)
        time_history.append(end_time - start_time)
        logger.info(f"[RL] delta: {delta}, reward: {reward}, time: {time_history[-1]:.4f}s")
        
        sub_result = env.bundle[-1]
        x_new = sub_result["pi"]
        f_new = sub_result["phi"]
        g_new = sub_result["g"]
        
        # 检查是否满足终止条件
        if stop_flag:
            logger.info(f"Warmstart - RL阶段满足终止条件，delta: {delta}")
            return delta_history, reward_history, time_history, switch_step
        
        # 检查 delta 是否不再变化（用于判断是否切换到 baseline）
        if len(delta_history) >= 2:
            delta_change = abs(delta_history[-1] - delta_history[-2])
            if delta_change < warmstart_threshold:
                consecutive_small_changes += 1
                logger.info(f"Warmstart - delta变化: {delta_change}, 连续次数: {consecutive_small_changes}")
                if consecutive_small_changes >= patience:
                    logger.info(f"Warmstart - delta连续{patience}次变化小于阈值，切换到baseline模式")
                    switch_step = step + 1
                    break
            else:
                consecutive_small_changes = 0
    
    # 如果切换到 baseline 模式
    if switch_step is not None:
        logger.info("==== SWITCHING TO BASELINE ====")
        for step in range(20 - switch_step):
            start_time = time.time()
            master.add_cut(x_new, f_new, g_new)
            ub, x_new = master.solve_master()
            g_new, f_new = env.subproblem.solve(x_new)
            serious_step, delta, stop_flag = master.update_strategy(x_new, f_new, g_new, ub)
            end_time = time.time()
            
            delta_history.append(delta)
            time_history.append(end_time - start_time)
            logger.info(f"[Baseline] delta: {delta}, time: {time_history[-1]:.4f}s")
            
            if stop_flag:
                logger.info(f"Warmstart - Baseline阶段满足终止条件，delta: {delta}")
                break

    logger.info("Warmstart test finished successfully.")
    
    return delta_history, reward_history, time_history, switch_step


def save_results(experiment_name, rl_delta, rl_reward, rl_time, baseline_delta, baseline_time, warmstart_delta=None, warmstart_time=None, switch_step=None):
    """保存结果到 JSON 文件"""
    results = {
        "experiment_name": experiment_name,
        "baseline": {
            "delta_history": [float(d) for d in baseline_delta],
            "time_history": [float(t) for t in baseline_time],
            "total_time": float(sum(baseline_time))
        },
        "rl_model": {
            "delta_history": [float(d) for d in rl_delta],
            "reward_history": [float(r) for r in rl_reward],
            "time_history": [float(t) for t in rl_time],
            "total_time": float(sum(rl_time))
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if warmstart_delta is not None:
        results["warmstart"] = {
            "delta_history": [float(d) for d in warmstart_delta],
            "time_history": [float(t) for t in warmstart_time],
            "total_time": float(sum(warmstart_time)),
            "switch_step": switch_step
        }
    
    log_dir = os.path.join("log", experiment_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    file_path = os.path.join(log_dir, "results.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"Results saved to: {file_path}")


def plot_results(rl_delta, rl_reward, baseline_delta, warmstart_delta=None, switch_step=None):
    """绘制对比结果图"""
    
    # 第一张图：RL 与 Baseline 对比
    plt.figure(figsize=(8, 5))
    plt.plot(rl_delta, marker='o', color='b', label='RL Model')
    plt.plot(baseline_delta, marker='s', color='r', label='Traditional Bundle')
    plt.xlabel('Iteration Step')
    plt.ylabel('Delta Value')
    plt.title('Convergence Comparison (RL vs Baseline)')
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 第二张图：Warmstart 与 Baseline 对比（如果有warmstart数据）
    if warmstart_delta is not None:
        plt.figure(figsize=(8, 5))
        # 绘制 baseline
        plt.plot(baseline_delta, marker='s', color='r', label='Traditional Bundle')
        
        # 绘制 warmstart，区分 RL 阶段和 Baseline 阶段
        if switch_step is not None and switch_step > 0:
            # RL 阶段（切换前）
            plt.plot(
                range(switch_step), 
                warmstart_delta[:switch_step], 
                marker='^', color='g', label='Warmstart (RL Phase)'
            )
            # Baseline 阶段（切换后）
            plt.plot(
                range(switch_step, len(warmstart_delta)), 
                warmstart_delta[switch_step:], 
                marker='v', color='orange', label='Warmstart (Baseline Phase)'
            )
        else:
            # 没有切换，全部是 RL 阶段
            plt.plot(warmstart_delta, marker='^', color='g', label='Warmstart (RL Phase)')
        
        plt.xlabel('Iteration Step')
        plt.ylabel('Delta Value')
        plt.title('Convergence Comparison (Warmstart vs Baseline)')
        plt.grid(True, alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    # 第三张图：Reward 变化图（仅 RL 模型）
    plt.figure(figsize=(8, 5))
    plt.plot(rl_reward, marker='s', color='r', label='Step Reward')
    plt.xlabel('Iteration Step')
    plt.ylabel('Reward')
    plt.title('Reward during Rollout (RL Model)')
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main(experiment_name, config, tolerance=1e-5, warmstart_threshold=1e-6, warmstart_patience=3):
    """加载最新训练的模型并进行测试"""
    import os
    from stable_baselines3 import PPO
    from model_train import SimpleBundleExtractor  # 导入自定义特征提取器
    log_dir = os.path.join("log", experiment_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = get_logger(os.path.join(log_dir, "bundle_env_test.log"))
    
    logger.info(f"实验参数: tolerance={tolerance}, warmstart_threshold={warmstart_threshold}, warmstart_patience={warmstart_patience}")

    # ===============================
    # 1️⃣ 实验配置
    # ===============================
    experiment_dir = os.path.join("model", experiment_name)

    # ===============================
    # 2️⃣ 找到最新的模型文件
    # ===============================
    model_files = []
    if os.path.exists(experiment_dir):
        for file in os.listdir(experiment_dir):
            if file.startswith("ppo_bundle_") and file.endswith(".zip"):
                model_files.append(file)

    if not model_files:
        logger.error(f"在 {experiment_dir} 中未找到模型文件")
        return

    # 按时间排序，取最新的
    model_files.sort(reverse=True)
    latest_model = model_files[0]
    model_path = os.path.join(experiment_dir, latest_model)
    logger.info(f"加载最新模型: {model_path}")

    # ===============================
    # 3️⃣ 加载模型（指定 custom_objects 以支持自定义特征提取器）
    # ===============================
    model = PPO.load(
        model_path,
        custom_objects={
            "SimpleBundleExtractor": SimpleBundleExtractor,
            "policy_kwargs": dict(
                features_extractor_class=SimpleBundleExtractor,
                features_extractor_kwargs=dict(features_dim=128),
                net_arch=dict(pi=[128, 128], vf=[128, 128])
            )
        }
    )


    # ===============================
    # 5️⃣ 使用传统 Bundle 算法求解作为 baseline
    # ===============================
    logger.info("==== Running Baseline (Traditional Bundle) ====")
    baseline_delta, baseline_time = bundle_baseline(logger, config, tolerance=tolerance)
    
    # ===============================
    # 6️⃣ 使用 RL 模型求解
    # ===============================
    logger.info("==== Running RL Model ====")
    test_env, test_master = create_env(logger, config, tolerance=tolerance)
    rl_delta, rl_reward, rl_time = bundle_RL(test_env, model, test_master, logger)
    
    # ===============================
    # 7️⃣ 使用 Warmstart 模式求解（RL + Baseline 混合）
    # ===============================
    logger.info("==== Running Warmstart Model ====")
    warmstart_env, warmstart_master = create_env(logger, config, tolerance=tolerance)
    warmstart_delta, warmstart_reward, warmstart_time, switch_step = bundle_RL_warmstart(
        warmstart_env, model, warmstart_master, logger,
        warmstart_threshold=warmstart_threshold,
        patience=warmstart_patience
    )
    logger.info(f"Warmstart - 切换步骤: {switch_step}")
    
    # ===============================
    # 8️⃣ 保存结果到 JSON 文件
    # ===============================
    save_results(
        experiment_name, rl_delta, rl_reward, rl_time,
        baseline_delta, baseline_time,
        warmstart_delta, warmstart_time, switch_step
    )
    
    # ===============================
    # 9️⃣ 绘制对比结果
    # ===============================
    plot_results(rl_delta, rl_reward, baseline_delta, warmstart_delta, switch_step)

def loadConfig(i, t, n):

    # config = BundleConfig(
    #     T=5,
    #     N_VARS=13,
    #     X_TRIAL=[-0.0, 1.0, 1.0],
    #     Y_TRIAL=[0.0, 131.60809087723158, 45.0],
    #     X_BS_TRIAL=[[-0.0, 0.0], [1.0, 1.0], [1.0, 1.0]],
    #     SOC_TRIAL=[0.0],
    #     PATH=Path(r"..\data\01_test_cases\case6ww\t24_n06"),
    #     n=4,  # realization 索引
    # )

    config_path = Path(f"./configs/config_{i}_{t}_{n}.pkl")
    config = BundleConfig.from_pkl(config_path)
    return config



if __name__ == "__main__":
    experiment_name = "multi_config_exp_02"  # 实验名称，用于区分不同实验
    config = loadConfig(i=2, t=12, n=3)
    # 可自定义参数：tolerance(收敛阈值), warmstart_threshold(切换阈值), warmstart_patience(连续次数)
    main(
        experiment_name, 
        config,
        tolerance=100,
        warmstart_threshold=100,
        warmstart_patience=2
    )