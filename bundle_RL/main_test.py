import json
import os
import time
from pathlib import Path

from matplotlib import pyplot as plt
from stable_baselines3 import PPO

from bundle_RL.config import BundleConfig
from bundle_RL.script.logger import get_logger
from bundle_RL.script.test import bundle_baseline, bundle_RL, bundle_RL_warmstart
from bundle_RL.script.train import SimpleBundleExtractor  # 导入自定义特征提取器
from bundle_RL.script.utils import create_env


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

    log_dir = os.path.join("test_result", experiment_name)
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

def main(experiment_name, config, tolerance=1e-5, warmstart_threshold=1e-6, warmstart_patience=3, deterministic=True):
    """加载最新训练的模型并进行测试"""

    log_dir = os.path.join("test_result", experiment_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = get_logger(os.path.join(log_dir, "bundle_env_test.log"))

    logger.info(f"实验参数: tolerance={tolerance}, warmstart_threshold={warmstart_threshold}, warmstart_patience={warmstart_patience}")

    # ===============================
    # 1️⃣ 实验配置
    # ===============================
    experiment_dir = os.path.join("train_result/model", experiment_name)

    # ===============================
    # 2️⃣ 找到最新的模型文件（从 save 子目录加载）
    # ===============================
    model_files = []
    model_dir = os.path.join(experiment_dir, "save")
    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            if file.startswith("ppo_bundle_") and file.endswith(".zip"):
                model_files.append(file)

    if not model_files:
        logger.error(f"在 {model_dir} 中未找到模型文件")
        return

    # 按时间排序，取最新的
    model_files.sort(reverse=True)
    latest_model = model_files[0]
    model_path = os.path.join(model_dir, latest_model)
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
    baseline_delta, baseline_time, baseline_ub, baseline_f_best = bundle_baseline(logger, config, tolerance=tolerance)

    # ===============================
    # 6️⃣ 使用 RL 模型求解
    # ===============================
    logger.info("==== Running RL Model ====")
    test_env, test_master = create_env(logger, config, tolerance=tolerance, verbose=True)  # 测试时启用详细日志
    rl_delta, rl_reward, rl_time, rl_ub, rl_f_best = bundle_RL(test_env, model, test_master, logger, deterministic)

    # ===============================
    # 7️⃣ 使用 Warmstart 模式求解（RL + Baseline 混合）
    # ===============================
    logger.info("==== Running Warmstart Model ====")
    warmstart_env, warmstart_master = create_env(logger, config, tolerance=tolerance, verbose=True)  # 测试时启用详细日志
    warmstart_delta, warmstart_reward, warmstart_time, warmstart_ub, warmstart_f_best, switch_step = bundle_RL_warmstart(
        warmstart_env, model, warmstart_master, logger,
        warmstart_threshold=warmstart_threshold,
        patience=warmstart_patience,
        deterministic=deterministic
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
    config = loadConfig(i=2, t=11, n=4)
    # 可自定义参数：tolerance(收敛阈值), warmstart_threshold(切换阈值), warmstart_patience(连续次数)
    main(
        experiment_name,
        config,
        tolerance=1e-2,
        warmstart_threshold=1e-2,
        warmstart_patience=2,
        deterministic=True,  # 每次选择最优的动作
    )