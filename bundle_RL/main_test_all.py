import json
import os
import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter

from bundle_RL.config import BundleConfig
from bundle_RL.script.logger import get_logger

from bundle_RL.script.test import bundle_baseline, bundle_RL, bundle_RL_warmstart


def compute_relative_gap(baseline_f_best, rl_f_best, rl_warmstart_f_best):
    """
    计算相对 gap: rel_gap = (LR - B) / LR
    LR = baseline收敛后的最终 f_best（最优下界）
    B = 当前方法在各步的 f_best

    返回的 rel_gap 数组长度应与对应的 f_best 数组长度一致

    负数表示当前方法找到的解比 baseline 最终收敛值更好（B > LR），这是期望的结果。
    """
    # 获取 baseline 最终收敛值作为 LR（最优下界）
    lr = baseline_f_best[-1] if len(baseline_f_best) > 0 else 1.0

    if abs(lr) < 1e-12:
        lr = 1.0

    rel_gap_dict = {
        "baseline": [],
        "rl": [],
        "rl_warmstart": []
    }

    # 计算 baseline 的相对 gap（应单调递减到 0）
    for b in baseline_f_best:
        gap = (lr - b) / lr
        rel_gap_dict["baseline"].append(gap)

    # 计算 RL 的相对 gap（负数表示比 baseline 更好）
    for b in rl_f_best:
        gap = (lr - b) / lr
        rel_gap_dict["rl"].append(gap)

    # 计算 RL Warmstart 的相对 gap（负数表示比 baseline 更好）
    for b in rl_warmstart_f_best:
        gap = (lr - b) / lr
        rel_gap_dict["rl_warmstart"].append(gap)

    return rel_gap_dict, lr


def compute_average_results(all_results):
    """对所有 config 的结果求均值"""
    max_steps = 0
    for result in all_results:
        for method in ["baseline", "rl", "rl_warmstart"]:
            if "rel_gap" in result[method]:
                max_steps = max(max_steps, len(result[method]["rel_gap"]))

    avg_results = {
        "baseline": {"rel_gap": [], "time": []},
        "rl": {"rel_gap": [], "time": []},
        "rl_warmstart": {"rel_gap": [], "time": []}
    }

    for step in range(max_steps):
        for method in ["baseline", "rl", "rl_warmstart"]:
            rel_gap_sum = 0.0
            time_sum = 0.0
            count = 0

            for result in all_results:
                if method in result:
                    rel_gap_list = result[method].get("rel_gap", [])
                    time_list = result[method].get("time", [])

                    # 关键修复：如果配置已收敛（数组长度不够），使用最后一个值
                    if step < len(rel_gap_list):
                        rel_gap_sum += rel_gap_list[step]
                    elif len(rel_gap_list) > 0:
                        # 已收敛，使用最后一个收敛值（gap 应为 0 或接近 0）
                        rel_gap_sum += rel_gap_list[-1]

                    if step < len(time_list):
                        time_sum += time_list[step]

                    count += 1  # 每个配置都计入，无论是否已收敛

            if count > 0:
                avg_results[method]["rel_gap"].append(rel_gap_sum / count)
                avg_results[method]["time"].append(time_sum / count)

    return avg_results


def plot_results(avg_results, save_dir, experiment_name=None, tb_writer=None):
    """绘制收敛对比图，支持保存为图片和 TensorBoard 格式"""

    # 计算 y 轴范围
    all_rel_gaps = (avg_results["baseline"]["rel_gap"] +
                    avg_results["rl"]["rel_gap"] +
                    avg_results["rl_warmstart"]["rel_gap"])
    if all_rel_gaps:
        y_min = min(all_rel_gaps)
        y_max = max(all_rel_gaps) * 1.1  # 留出 10% 的上边距
    else:
        y_min, y_max = 0, 1

    # 图1：gap随迭代次数的收敛图
    plt.figure(figsize=(8, 5))
    # 使用同一颜色的不同深浅区分三个方法
    plt.plot(avg_results["baseline"]["rel_gap"], marker='s', color='#1a1a1a', label='Baseline', linewidth=2)  # 最深
    plt.plot(avg_results["rl"]["rel_gap"], marker='o', color='#666666', label='RL', linewidth=2)  # 中等
    plt.plot(avg_results["rl_warmstart"]["rel_gap"], marker='^', color='#b3b3b3', label='RL Warmstart',
             linewidth=2)  # 最浅
    plt.xlabel('Iteration Step')
    plt.ylabel('Relative Gap')
    plt.title('Convergence vs Iteration')
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.ylim(y_min, y_max)
    plt.tight_layout()

    # 保存图片
    conv_plot_path = os.path.join(save_dir, 'convergence_iteration.png')
    plt.savefig(conv_plot_path)

    # 添加到 TensorBoard
    if tb_writer is not None:
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        from PIL import Image
        image = Image.open(buf)
        tb_writer.add_image('convergence_iteration', np.array(image).transpose(2, 0, 1), 0)

    plt.close()

    # 图2：gap随运行时间的收敛图
    plt.figure(figsize=(8, 5))
    baseline_cum_time = np.cumsum(avg_results["baseline"]["time"])
    rl_cum_time = np.cumsum(avg_results["rl"]["time"])
    rl_warmstart_cum_time = np.cumsum(avg_results["rl_warmstart"]["time"])

    # 使用同一颜色的不同深浅区分三个方法
    plt.plot(baseline_cum_time, avg_results["baseline"]["rel_gap"], marker='s', color='#1a1a1a', label='Baseline',
             linewidth=2)  # 最深
    plt.plot(rl_cum_time, avg_results["rl"]["rel_gap"], marker='o', color='#666666', label='RL', linewidth=2)  # 中等
    plt.plot(rl_warmstart_cum_time, avg_results["rl_warmstart"]["rel_gap"], marker='^', color='#b3b3b3',
             label='RL Warmstart', linewidth=2)  # 最浅
    plt.xlabel('Cumulative Time (s)')
    plt.ylabel('Relative Gap')
    plt.title('Convergence vs Time')
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.ylim(y_min, y_max)
    plt.tight_layout()

    # 保存图片
    time_plot_path = os.path.join(save_dir, 'convergence_time.png')
    plt.savefig(time_plot_path)

    # 添加到 TensorBoard
    if tb_writer is not None:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        tb_writer.add_image('convergence_time', np.array(image).transpose(2, 0, 1), 0)

    plt.close()

    print(f"Convergence plots saved to {save_dir}")

    # 同时保存数值数据（用于对比分析）
    if tb_writer is not None:
        # 收敛曲线数据（用于 TensorBoard 的 scalar 对比）
        for step, gap in enumerate(avg_results["baseline"]["rel_gap"]):
            tb_writer.add_scalar('metrics/baseline_gap', gap, step)
        for step, gap in enumerate(avg_results["rl"]["rel_gap"]):
            tb_writer.add_scalar('metrics/rl_gap', gap, step)
        for step, gap in enumerate(avg_results["rl_warmstart"]["rel_gap"]):
            tb_writer.add_scalar('metrics/rl_warmstart_gap', gap, step)

        print(f"TensorBoard data saved")


def save_results_to_json(all_results, save_dir):
    """将所有结果保存为 JSON（追加写）"""
    results_file = os.path.join(save_dir, 'all_results.json')

    save_data = []
    for result in all_results:
        config_data = {
            "i": result["config_info"]["i"],
            "t": result["config_info"]["t"],
            "n": result["config_info"]["n"],
            "lr": result["lr"],
            "baseline": {
                "delta_history": [float(d) for d in result["baseline"].get("delta", [])],
                "time_history": [float(t) for t in result["baseline"].get("time", [])],
                "ub_history": [float(u) for u in result["baseline"].get("ub", [])],
                "f_best_history": [float(f) for f in result["baseline"].get("f_best", [])],
                "rel_gap": [float(r) for r in result["baseline"].get("rel_gap", [])]
            },
            "rl": {
                "delta_history": [float(d) for d in result["rl"].get("delta", [])],
                "time_history": [float(t) for t in result["rl"].get("time", [])],
                "ub_history": [float(u) for u in result["rl"].get("ub", [])],
                "f_best_history": [float(f) for f in result["rl"].get("f_best", [])],
                "rel_gap": [float(r) for r in result["rl"].get("rel_gap", [])],
                "switch_step": result["rl"].get("switch_step")
            },
            "rl_warmstart": {
                "delta_history": [float(d) for d in result["rl_warmstart"].get("delta", [])],
                "time_history": [float(t) for t in result["rl_warmstart"].get("time", [])],
                "ub_history": [float(u) for u in result["rl_warmstart"].get("ub", [])],
                "f_best_history": [float(f) for f in result["rl_warmstart"].get("f_best", [])],
                "rel_gap": [float(r) for r in result["rl_warmstart"].get("rel_gap", [])],
                "switch_step": result["rl_warmstart"].get("switch_step")
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        save_data.append(config_data)

    with open(results_file, 'a', encoding='utf-8') as f:
        for data in save_data:
            json.dump(data, f)
            f.write('\n')

    print(f"Results saved to {results_file}")


def load_latest_model(train_experiment_name, logger):
    """加载最新训练的模型"""
    model_dir = os.path.join("train_result", "model", train_experiment_name, "save")

    if not os.path.exists(model_dir):
        error_msg = f"模型目录不存在: {model_dir}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    model_files = [f for f in os.listdir(model_dir)
                   if f.startswith("ppo_bundle_") and f.endswith(".zip")]

    if not model_files:
        error_msg = f"在 {model_dir} 中未找到模型文件"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    latest_model = sorted(model_files, reverse=True)[0]
    model_path = os.path.join(model_dir, latest_model)
    logger.info(f"加载模型: {model_path}")

    try:
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
        return model
    except Exception as e:
        error_msg = f"模型加载失败: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def run_test_for_configs(configs, config_info_list, experiment_name, logger, model,
                         tolerance=1e-5, warmstart_threshold=1e-4, patience=3, K=20):
    """
    对一组 config 运行测试，计算相对 gap
    """
    all_results = []

    for idx, (config, config_info) in enumerate(zip(configs, config_info_list)):
        logger.info(
            f"=== Testing config {idx + 1}/{len(configs)}: i={config_info['i']}, t={config_info['t']}, n={config_info['n']} ===")

        # 1. 运行 baseline
        logger.info("Running Baseline...")
        baseline_delta, baseline_time, baseline_ub, baseline_f_best = bundle_baseline(logger, config,
                                                                                      tolerance=tolerance)

        # 2. 运行 RL
        logger.info("Running RL...")
        test_env, test_master = BundleDualEnv.create_env(logger, config, tolerance=tolerance, verbose=True, K=K)  # 测试时启用详细日志
        rl_delta, rl_reward, rl_time, rl_ub, rl_f_best = bundle_RL(
            test_env, model, test_master, logger, deterministic=True)
        rl_switch_step = None

        # 3. 运行 RL Warmstart
        logger.info("Running RL Warmstart...")
        warmstart_env, warmstart_master = BundleDualEnv.create_env(logger, config, tolerance=tolerance, verbose=True,
                                                     K=K)  # 测试时启用详细日志
        ws_delta, ws_reward, ws_time, ws_ub, ws_f_best, ws_switch_step = bundle_RL_warmstart(
            warmstart_env, model, warmstart_master, logger,
            warmstart_threshold=warmstart_threshold,
            patience=patience,
            deterministic=True)

        # 4. 计算相对 gap
        rel_gap_dict, lr = compute_relative_gap(baseline_f_best, rl_f_best, ws_f_best)

        # 5. 保存结果
        result = {
            "config_info": config_info,
            "lr": lr,
            "baseline": {
                "delta": baseline_delta,
                "time": baseline_time,
                "ub": baseline_ub,
                "f_best": baseline_f_best,
                "rel_gap": rel_gap_dict["baseline"]
            },
            "rl": {
                "delta": rl_delta,
                "time": rl_time,
                "ub": rl_ub,
                "f_best": rl_f_best,
                "rel_gap": rel_gap_dict["rl"],
                "switch_step": rl_switch_step
            },
            "rl_warmstart": {
                "delta": ws_delta,
                "time": ws_time,
                "ub": ws_ub,
                "f_best": ws_f_best,
                "rel_gap": rel_gap_dict["rl_warmstart"],
                "switch_step": ws_switch_step
            }
        }

        all_results.append(result)
        logger.info(f"=== Finished config {idx + 1}/{len(configs)} ===")

    return all_results


def main(experiment_name, train_experiment_name=None, i=2, tolerance=1e-5, warmstart_threshold=1e-4, patience=3, K=20,
         tb_writer=None):
    """主测试函数"""
    if train_experiment_name is None:
        train_experiment_name = experiment_name

    save_dir = os.path.join("test_result", experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    logger = get_logger(os.path.join(save_dir, "test_all.log"))

    # 加载训练好的模型（失败时直接抛出异常）
    logger.info(f"加载模型: {train_experiment_name}")
    model = load_latest_model(train_experiment_name, logger)

    # 收集 configs
    logger.info(f"Collecting configs for i={i}...")
    configs, config_info_list = collect_configs(i=i)
    logger.info(f"Loaded {len(configs)} configs")

    if not configs:
        logger.error("No configs found!")
        return

    # 运行测试
    logger.info("Running tests for all configs...")
    all_results = run_test_for_configs(
        configs,
        config_info_list,
        experiment_name,
        logger,
        model,
        tolerance=tolerance,
        warmstart_threshold=warmstart_threshold,
        patience=patience,
        K=K
    )

    # 计算平均结果
    logger.info("Computing average results...")
    avg_results = compute_average_results(all_results)

    # 保存结果
    logger.info("Saving results...")
    save_results_to_json(all_results, save_dir)

    # 绘制图表（支持 TensorBoard）
    logger.info("Plotting results...")
    plot_results(avg_results, save_dir, experiment_name=experiment_name, tb_writer=tb_writer)

    logger.info("All tests completed!")


def collect_configs(i=2):
    """收集指定 i 的所有 config"""
    configs = []
    config_info = []

    for t in range(1, 24):
        for n in range(1):
            config_path = Path(f"./configs/config_{i}_{t}_{n}.pkl")
            if config_path.exists():
                config = BundleConfig.from_pkl(config_path)
                configs.append(config)
                config_info.append({"i": i, "t": t, "n": n})
                print(f"Loaded config_{i}_{t}_{n}.pkl")
            else:
                print(f"Config file not found: {config_path}")

    return configs, config_info


if __name__ == "__main__":

    from bundle_RL.script.default_feature.env import BundleDualEnv
    from bundle_RL.script.default_feature.train import SimpleBundleExtractor

    # ==============================================
    # 批量测试配置列表
    # ==============================================
    test_configs = [
        # (experiment_name, train_experiment_name, K)
    ]
    for i in range(25, 26):
        test_configs.append(
            (f"exp{i:02d}", f"exp{i:02d}", 10)
        )

    # ==============================================
    # TensorBoard 根目录（所有 runs 的父目录）
    # ==============================================
    tb_root_dir = os.path.join("test_result", "tb_summary")
    os.makedirs(tb_root_dir, exist_ok=True)
    print(f"TensorBoard 根目录: {tb_root_dir}")
    print(f"运行命令查看: tensorboard --logdir={tb_root_dir}")

    # ==============================================
    # 遍历所有配置进行批量测试（每个测试作为独立的 TensorBoard run）
    # ==============================================
    for idx, (exp_name, train_exp_name, K_val) in enumerate(test_configs):
        # 每个测试作为独立的 run，命名格式为 exp{i:02d}
        run_name = f"exp{idx + 16:02d}"  # 从 exp16 开始
        tb_log_dir = os.path.join(tb_root_dir, run_name)

        print(f"\n{'=' * 60}")
        print(f"开始测试: {exp_name}")
        print(f"训练模型: {train_exp_name}, K={K_val}")
        print(f"TensorBoard Run: {run_name}")
        print(f"{'=' * 60}")

        # 为每个 run 创建独立的 SummaryWriter
        tb_writer = SummaryWriter(log_dir=tb_log_dir)

        try:
            main(
                experiment_name=exp_name,  # 测试结果保存目录名
                train_experiment_name=train_exp_name,  # 训练模型所在的实验名
                i=3,
                tolerance=1e-3,
                warmstart_threshold=0.01,
                patience=3,
                K=K_val,  # 必须与训练时的 K 值一致！
                tb_writer=tb_writer  # 传递独立的 TensorBoard writer
            )
            print(f"\n✅ {exp_name} 测试完成!")
        except Exception as e:
            print(f"\n❌ {exp_name} 测试失败: {str(e)}")
            import traceback

            traceback.print_exc()

        # 关闭当前 run 的 writer
        tb_writer.close()
        print(f"\n{'=' * 60}")

    print(f"\n{'=' * 60}")
    print(f"所有测试完成!")
    print(f"TensorBoard 日志已保存到: {tb_root_dir}")
    print(f"运行以下命令查看:")
    print(f"  tensorboard --logdir={tb_root_dir}")