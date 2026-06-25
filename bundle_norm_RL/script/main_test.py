"""
Level Bundle RL 测试入口

遵循 bundle_RL/script/attention2/main_test.py 的代码风格:
- 加载训练好的模型
- 对比 baseline (Level Bundle) vs RL
- 保存结果到 JSON 和图片
- TensorBoard 日志
"""

import json
import os
import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3 import PPO

from bundle_norm_RL.script.config import LevelBundleConfig
from bundle_norm_RL.script.env import LevelBundleEnv
from bundle_norm_RL.script.features_extractor import LevelBundleFeaturesExtractor
from bundle_norm_RL.script.policy_network import LevelBundleActorCriticPolicy
from bundle_norm_RL.script.level_bundle_problem import LevelBundleSolver
from bundle_norm_RL.script.logger import get_logger


def level_bundle_baseline(logger, config, n=0):
    """
    运行 Level Bundle baseline

    Returns:
        lb_history, time_history, ub_history, f_best_history
    """
    solver = LevelBundleSolver(logger, config, n=n)
    results = solver.solve()

    return results.lb, results.solver_time, results.ub, results.lb


def level_bundle_rl(env, model, logger, deterministic=True, K=20):
    """
    使用 RL 模型预测乘子

    Returns:
        lb_history, reward_history, time_history
    """
    obs, _ = env.reset()
    lb_history = []
    reward_history = []
    time_history = []

    for step in range(K):
        t0 = time.time()
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        elapsed = time.time() - t0

        lb_history.append(env.LB)
        reward_history.append(reward)
        time_history.append(elapsed)

        if terminated or truncated:
            break

    return lb_history, reward_history, time_history


def load_latest_model(train_experiment_name, logger, hidden_dim=64):
    """加载最新训练的模型"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)  # bundle_norm_RL
    base_dir = str(base_dir)

    model_dir = os.path.join(base_dir, "train_result", "model", train_experiment_name, "save")

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")

    model_files = [f for f in os.listdir(model_dir)
                   if f.startswith("ppo_level_bundle_") and f.endswith(".zip")]

    if not model_files:
        raise FileNotFoundError(f"在 {model_dir} 中未找到模型文件")

    latest_model = sorted(model_files, reverse=True)[0]
    model_path = os.path.join(model_dir, latest_model)
    logger.info(f"加载模型: {model_path}")

    model = PPO.load(
        model_path,
        custom_objects={
            "LevelBundleFeaturesExtractor": LevelBundleFeaturesExtractor,
            "LevelBundleActorCriticPolicy": LevelBundleActorCriticPolicy,
        }
    )
    return model


def collect_configs(i=1, t=5):
    """收集指定 i, t 的所有 config"""
    configs = []
    config_info = []

    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)  # bundle_norm_RL
    config_dir = Path(os.path.join(base_dir, "configs"))

    for n in range(6):
        config_path = config_dir / f"config_{i}_{t}_{n}.pkl"
        if config_path.exists():
            config = LevelBundleConfig.from_pkl(config_path)
            configs.append(config)
            config_info.append({"i": i, "t": t, "n": n})
            print(f"Loaded config_{i}_{t}_{n}.pkl")
        else:
            print(f"Config file not found: {config_path}")

    return configs, config_info


def save_results_to_json(all_results, save_dir):
    """将所有结果保存为 JSON"""
    results_file = os.path.join(save_dir, 'all_results.json')

    save_data = []
    for result in all_results:
        data = {
            "config_info": result["config_info"],
            "baseline": {
                "lb_history": [float(x) for x in result.get("baseline", {}).get("lb_history", [])],
                "time_history": [float(x) for x in result.get("baseline", {}).get("time_history", [])],
            },
            "rl": {
                "lb_history": [float(x) for x in result["rl"].get("lb_history", [])],
                "reward_history": [float(x) for x in result["rl"].get("reward_history", [])],
                "time_history": [float(x) for x in result["rl"].get("time_history", [])],
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_data.append(data)

    with open(results_file, 'a', encoding='utf-8') as f:
        for data in save_data:
            json.dump(data, f)
            f.write('\n')

    print(f"Results saved to {results_file}")


def plot_results(all_results, save_dir):
    """绘制 LB 收敛对比图"""
    if not all_results:
        return

    plt.figure(figsize=(10, 6))

    for idx, result in enumerate(all_results):
        rl_lb = result["rl"].get("lb_history", [])
        if rl_lb:
            plt.plot(rl_lb, alpha=0.5, label=f"RL n={result['config_info']['n']}")

    plt.xlabel('Step')
    plt.ylabel('Lower Bound')
    plt.title('Level Bundle RL: LB Convergence')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(save_dir, 'lb_convergence.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {plot_path}")


def main(experiment_name, train_experiment_name=None, i=1, t=5, K=20, hidden_dim=64):
    """主测试函数"""
    if train_experiment_name is None:
        train_experiment_name = experiment_name

    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)  # bundle_norm_RL
    base_dir = str(base_dir)

    save_dir = os.path.join(base_dir, "test_result", experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    logger = get_logger(os.path.join(save_dir, "test.log"))

    logger.info(f"加载模型: {train_experiment_name}")
    model = load_latest_model(train_experiment_name, logger, hidden_dim=hidden_dim)

    logger.info(f"Collecting configs for i={i}, t={t}...")
    configs, config_info_list = collect_configs(i=i, t=t)
    logger.info(f"Loaded {len(configs)} configs")

    if not configs:
        logger.error("No configs found!")
        return

    all_results = []

    for idx, (config, config_info) in enumerate(zip(configs, config_info_list)):
        logger.info(f"=== Testing config {idx+1}/{len(configs)}: i={config_info['i']}, t={config_info['t']}, n={config_info['n']} ===")

        # RL 测试
        logger.info("Running RL...")
        env = LevelBundleEnv.create_env(logger, config, K=K, verbose=True, use_outer=False)
        rl_lb, rl_reward, rl_time = level_bundle_rl(env, model, logger, deterministic=True, K=K)
        env.close()

        result = {
            "config_info": config_info,
            "rl": {
                "lb_history": rl_lb,
                "reward_history": rl_reward,
                "time_history": rl_time,
            },
        }

        all_results.append(result)
        logger.info(f"=== Finished config {idx+1}/{len(configs)} ===")

    logger.info("Saving results...")
    save_results_to_json(all_results, save_dir)

    logger.info("Plotting results...")
    plot_results(all_results, save_dir)

    logger.info("All tests completed!")


if __name__ == "__main__":
    experiment_name = "exp_06"
    main(
        experiment_name=experiment_name,
        train_experiment_name=experiment_name,
        i=1,
        t=5,
        K=20,
        hidden_dim=64,
    )
