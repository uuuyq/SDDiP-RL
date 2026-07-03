"""
Incremental Level Bundle RL 测试入口

与 script/main_test.py 保持相同风格，支持三种对比:
1. Baseline: 传统增量 Level Bundle 算法
2. RL: 纯 RL 模型预测增量
3. RL Warmstart: RL 阶段 + 当 gap 不再下降时切换到 baseline
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import yaml
from matplotlib import pyplot as plt
from stable_baselines3 import PPO

from bundle_norm_RL.script1.config import LevelBundleConfig
from bundle_norm_RL.script1.env import IncrementalLevelBundleEnv
from bundle_norm_RL.script1.features_extractor import IncrementalLevelBundleFeaturesExtractor
from bundle_norm_RL.script1.policy_network import IncrementalLevelBundleActorCriticPolicy
from bundle_norm_RL.script1.level_bundle_problem import LevelBundleSolver, IncrementalOuterProblem, InnerProblem
from bundle_norm_RL.script1.logger import get_logger


# ============================================================
# 测试函数
# ============================================================

def level_bundle_baseline(logger, config, n=0):
    """
    运行增量 Level Bundle baseline

    Returns:
        final_lb, final_ub, solve_time, lb_history, ub_history, time_history
    """
    solver = LevelBundleSolver(logger, config, n=n)
    results = solver.solve()
    return (results.lb, results.ub, results.solver_time,
            results.lb_history, results.ub_history, results.time_history)


def level_bundle_rl(env, model, logger, deterministic=True, K=20):
    """
    使用 RL 模型预测增量，同时用 IncrementalOuterProblem 跟踪 UB

    Returns:
        lb_history, reward_history, time_history, ub_history, f_best_history
    """
    obs, _ = env.reset()

    lb_history = []
    reward_history = []
    time_history = []
    ub_history = []
    f_best_history = []

    # 创建独立的 IncrementalOuterProblem 用于评估 UB
    outer = IncrementalOuterProblem(
        logger,
        dim_pi=env.config.N_VARS,
        X_trial=env.config.X_trial,
        theta_trial=float(env.config.THETA_TRIAL),
        rho=env.config.rho,
        B_t=env.config.B_t,
        norm_bound_type=env.config.norm_bound_type,
        weights=env.config.weights,
    )

    # 添加初始次梯度
    if len(env.subgradient_list) > 0:
        lambda_i = np.concatenate([env.pi, [env.pi0]])
        omega_i = env.LB
        outer.add_cut(env.subgradient_list[0].tolist(), lambda_i, omega_i)
        _, _, ub = outer.solve()
        if ub is not None:
            ub_history.append(ub)
        else:
            ub_history.append(env.LB)

    f_best_history.append(env.LB)
    lb_history.append(env.current_dual)

    for step in range(K):
        t0 = time.time()
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        elapsed = time.time() - t0

        # 添加新的次梯度到 outer problem
        if len(env.subgradient_list) > step + 1:
            lambda_i = np.concatenate([env.pi, [env.pi0]])
            omega_i = env.current_dual
            outer.add_cut(env.subgradient_list[step + 1].tolist(), lambda_i, omega_i)
            _, _, ub = outer.solve()
            if ub is not None:
                ub = min(ub, ub_history[-1]) if ub_history else ub
                ub_history.append(ub)
            else:
                ub_history.append(ub_history[-1] if ub_history else env.LB)
        else:
            ub_history.append(ub_history[-1] if ub_history else env.LB)

        lb_history.append(env.current_dual)
        reward_history.append(reward)
        time_history.append(elapsed)
        f_best_history.append(env.LB)

        logger.info(
            f"RL step {step+1}: LB={env.LB:.6f}, UB={ub_history[-1]:.6f}, "
            f"gap={ub_history[-1]-env.LB:.6e}, reward={reward:.6f}"
        )

        if terminated or truncated:
            break

    return lb_history, reward_history, time_history, ub_history, f_best_history


def level_bundle_rl_warmstart(env, model, logger, deterministic=True, K=20,
                               warmstart_threshold=1e-4, patience=3):
    """
    Warm-start 测试: RL 阶段 + 当 gap 不再下降时切换到 baseline

    Returns:
        lb_history, reward_history, time_history, ub_history, f_best_history, switch_step
    """
    obs, _ = env.reset()

    lb_history = []
    reward_history = []
    time_history = []
    ub_history = []
    f_best_history = []
    switch_step = None
    consecutive_small_changes = 0

    # 创建独立的 IncrementalOuterProblem
    outer = IncrementalOuterProblem(
        logger,
        dim_pi=env.config.N_VARS,
        X_trial=env.config.X_trial,
        theta_trial=float(env.config.THETA_TRIAL),
        rho=env.config.rho,
        B_t=env.config.B_t,
        norm_bound_type=env.config.norm_bound_type,
        weights=env.config.weights,
    )

    if len(env.subgradient_list) > 0:
        lambda_i = np.concatenate([env.pi, [env.pi0]])
        omega_i = env.LB
        outer.add_cut(env.subgradient_list[0].tolist(), lambda_i, omega_i)
        _, _, ub = outer.solve()
        if ub is not None:
            ub_history.append(ub)
        else:
            ub_history.append(env.LB)

    f_best_history.append(env.LB)
    lb_history.append(env.current_dual)

    # ===== RL 阶段 =====
    for step in range(K):
        t0 = time.time()
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        elapsed = time.time() - t0

        if len(env.subgradient_list) > step + 1:
            lambda_i = np.concatenate([env.pi, [env.pi0]])
            omega_i = env.current_dual
            outer.add_cut(env.subgradient_list[step + 1].tolist(), lambda_i, omega_i)
            _, _, ub = outer.solve()
            if ub is not None:
                ub = min(ub, ub_history[-1]) if ub_history else ub
                ub_history.append(ub)
            else:
                ub_history.append(ub_history[-1] if ub_history else env.LB)
        else:
            ub_history.append(ub_history[-1] if ub_history else env.LB)

        lb_history.append(env.current_dual)
        reward_history.append(reward)
        time_history.append(elapsed)
        f_best_history.append(env.LB)

        current_gap = ub_history[-1] - env.LB
        logger.info(
            f"[RL] step {step+1}: LB={env.LB:.6f}, UB={ub_history[-1]:.6f}, "
            f"gap={current_gap:.6e}, reward={reward:.6f}"
        )

        if terminated or truncated:
            break

        # 检查 gap 是否不再下降
        if len(lb_history) >= 2:
            gap_change = abs(lb_history[-1] - lb_history[-2])
            gap_ref = max(abs(lb_history[-1]), 1)
            relative_change = gap_change / gap_ref

            if lb_history[-1] < lb_history[-2]:
                logger.info(
                    f"Warmstart - LB 反向下降: {lb_history[-2]:.6f} -> {lb_history[-1]:.6f}，"
                    f"切换到 baseline"
                )
                switch_step = step + 1
                break
            elif relative_change < warmstart_threshold:
                consecutive_small_changes += 1
                logger.info(
                    f"Warmstart - LB 相对变化: {relative_change:.6e}, "
                    f"连续次数: {consecutive_small_changes}/{patience}"
                )
                if consecutive_small_changes >= patience:
                    logger.info(
                        f"Warmstart - LB 连续 {patience} 次下降不明显，切换到 baseline"
                    )
                    switch_step = step + 1
                    break
            else:
                consecutive_small_changes = 0

    # ===== Baseline 阶段 =====
    if switch_step is not None:
        logger.info("==== SWITCHING TO BASELINE ====")
        remaining_steps = K - switch_step

        pi_hat = env.pi.copy()
        pi0_hat = env.pi0
        pi_bar = env.pi_bar.copy()
        pi0_bar = env.pi0_bar

        # 同步稳定中心
        outer.set_stability_center(pi_bar, pi0_bar)

        X_trial = env.X_trial
        theta_trial = env.theta_trial

        for step in range(remaining_steps):
            t0 = time.time()

            z_X_values, obj_term_value, inner_obj = env.inner_problem.solve(pi_hat, pi0_hat)
            if z_X_values is None:
                break

            subgradient = z_X_values + [obj_term_value]
            lambda_i = np.concatenate([pi_hat, [pi0_hat]])
            omega_i = inner_obj - pi_hat @ X_trial - pi0_hat * theta_trial
            outer.add_cut(subgradient, lambda_i, omega_i)

            pi_dummy, pi0_dummy, outer_obj = outer.solve()
            if outer_obj is None:
                break

            dual = inner_obj - pi_hat @ X_trial - pi0_hat * theta_trial
            if dual > env.LB:
                env.LB = dual

            ub = outer_obj

            elapsed = time.time() - t0
            lb_history.append(dual)
            time_history.append(elapsed)
            ub = min(ub, ub_history[-1]) if ub_history else ub
            ub_history.append(ub)
            f_best_history.append(env.LB)

            logger.info(
                f"[Baseline] step {step+1}: LB={env.LB:.6f}, UB={ub:.6f}, "
                f"gap={ub-env.LB:.6e}"
            )

            if ub - env.LB < env.config.gap_tol * abs(ub) or ub - env.LB < 1e-6:
                break

            # Level 策略
            level = ub - env.config.level_factor * (ub - env.LB)
            outer.set_level(level)
            outer.model.params.Method = 2
            outer.model.update()
            outer.model.optimize()

            if outer.model.status != 2:
                outer.model.params.Method = 1
                outer.model.update()
                outer.model.optimize()
                if outer.model.status != 2:
                    outer.model.params.Method = 0
                    outer.model.update()
                    outer.model.optimize()
                    if outer.model.status != 2:
                        outer.recover()
                        pi_hat = np.array(pi_dummy)
                        pi0_hat = pi0_dummy
                        continue

            d_pi_star = np.array([outer.d_pi[i].x for i in range(env.config.N_VARS)])
            d_pi0_star = outer.d_pi0.x
            pi_hat = outer.pi_bar + d_pi_star
            pi0_hat = outer.pi0_bar + d_pi0_star
            outer.recover()

            # Serious step
            current_dual = inner_obj - pi_hat @ X_trial - pi0_hat * theta_trial
            if current_dual > env.LB + 1e-8 * max(1.0, abs(env.LB)):
                pi_bar = pi_hat.copy()
                pi0_bar = pi0_hat
                outer.set_stability_center(pi_bar, pi0_bar)

    return lb_history, reward_history, time_history, ub_history, f_best_history, switch_step


# ============================================================
# 结果分析
# ============================================================

def compute_opt_gap(lb_history, ub_history):
    """计算归一化优化 gap: (UB - LB) / (UB_0 - LB_0)"""
    if len(lb_history) == 0 or len(ub_history) == 0:
        return []
    initial_gap = ub_history[0] - lb_history[0]
    if abs(initial_gap) < 1e-12:
        return [0.0] * len(lb_history)
    return [(ub - lb) / initial_gap for lb, ub in zip(lb_history, ub_history)]


def compute_average_results(all_results):
    """对所有 config 的 baseline/RL/RL Warmstart 的 gap 求均值"""
    valid_indices = []
    for idx, result in enumerate(all_results):
        baseline_gap = result.get("baseline", {}).get("gap", [])
        if len(baseline_gap) > 1:
            valid_indices.append(idx)
        else:
            config_info = result.get("config_info", {})
            print(f"排除首次迭代即收敛的 config: i={config_info.get('i')}, "
                  f"t={config_info.get('t')}, n={config_info.get('n')}, "
                  f"baseline_gap_len={len(baseline_gap)}")

    if not valid_indices:
        print("警告：所有 config 均首次迭代即收敛，无有效数据")
        return {
            "baseline": {"gap": [], "time": []},
            "rl": {"gap": [], "time": []},
            "rl_warmstart": {"gap": [], "time": []},
        }

    filtered_results = [all_results[i] for i in valid_indices]

    max_steps = 0
    for result in filtered_results:
        for method in ["baseline", "rl", "rl_warmstart"]:
            if "gap" in result[method]:
                max_steps = max(max_steps, len(result[method]["gap"]))

    avg_results = {
        "baseline": {"gap": [], "time": []},
        "rl": {"gap": [], "time": []},
        "rl_warmstart": {"gap": [], "time": []},
    }

    for step in range(max_steps):
        for method in ["baseline", "rl", "rl_warmstart"]:
            gap_sum = 0.0
            time_sum = 0.0
            count = 0

            for result in filtered_results:
                if method in result:
                    gap_list = result[method].get("gap", [])
                    time_list = result[method].get("time", [])

                    if step < len(gap_list):
                        gap_sum += gap_list[step]
                        count += 1

                    if step < len(time_list):
                        time_sum += time_list[step]

            if count > 0:
                avg_results[method]["gap"].append(gap_sum / count)
                avg_results[method]["time"].append(time_sum / count)

    print(f"参与均值计算的 config 数: {len(filtered_results)}/{len(all_results)}")
    return avg_results


# ============================================================
# 绘图
# ============================================================

def plot_results(avg_results, save_dir, experiment_name=None):
    """绘制收敛对比图"""
    all_gaps = (avg_results["baseline"]["gap"]
                + avg_results["rl"]["gap"]
                + avg_results["rl_warmstart"]["gap"])
    if all_gaps:
        y_min = min(0, min(all_gaps))
        y_max = max(all_gaps) * 1.1
    else:
        y_min, y_max = 0, 1

    # 图1: gap vs 迭代次数
    plt.figure(figsize=(8, 5))
    plt.plot(avg_results["baseline"]["gap"], marker='s', color='#2196F3',
             label='Baseline', linewidth=2)
    plt.plot(avg_results["rl"]["gap"], marker='o', color='#FF5722',
             label='RL', linewidth=2)
    plt.plot(avg_results["rl_warmstart"]["gap"], marker='^', color='#4CAF50',
             label='RL Warmstart', linewidth=2)
    plt.xlabel('Iteration Step')
    plt.ylabel('Optimization Gap: (UB - LB) / |UB|')
    plt.title(f'Convergence vs Iteration ({experiment_name or ""})')
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'convergence_iteration.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 图2: gap vs 累计时间
    plt.figure(figsize=(8, 5))
    baseline_cum_time = np.cumsum(avg_results["baseline"]["time"])
    rl_cum_time = np.cumsum(avg_results["rl"]["time"])
    rl_warmstart_cum_time = np.cumsum(avg_results["rl_warmstart"]["time"])

    plt.plot(baseline_cum_time, avg_results["baseline"]["gap"], marker='s',
             color='#2196F3', label='Baseline', linewidth=2)
    plt.plot(rl_cum_time, avg_results["rl"]["gap"], marker='o',
             color='#FF5722', label='RL', linewidth=2)
    plt.plot(rl_warmstart_cum_time, avg_results["rl_warmstart"]["gap"], marker='^',
             color='#4CAF50', label='RL Warmstart', linewidth=2)
    plt.xlabel('Cumulative Time (s)')
    plt.ylabel('Optimization Gap: (UB - LB) / |UB|')
    plt.title(f'Convergence vs Time ({experiment_name or ""})')
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'convergence_time.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Convergence plots saved to {save_dir}")


# ============================================================
# 模型加载
# ============================================================

def load_train_config(train_experiment_name):
    """从训练实验目录加载训练时保存的配置文件"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    config_path = os.path.join(base_dir, "train_result", "model", train_experiment_name,
                               f"{train_experiment_name}.yml")

    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"加载训练配置: {config_path}")
        return config
    else:
        print(f"训练配置文件不存在: {config_path}，使用默认配置")
        return None


def load_latest_model(train_experiment_name, logger,
                      hidden_dim=128, encoder_type="self_attention",
                      n_heads=4, n_attn_layers=2):
    """加载最新训练的模型"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    base_dir = str(base_dir)

    model_dir = os.path.join(base_dir, "train_result", "model", train_experiment_name, "save")

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")

    model_files = [f for f in os.listdir(model_dir)
                   if f.startswith("ppo_incremental_level_bundle_") and f.endswith(".zip")]

    if not model_files:
        raise FileNotFoundError(f"在 {model_dir} 中未找到模型文件")

    latest_model = sorted(model_files, reverse=True)[0]
    model_path = os.path.join(model_dir, latest_model)
    logger.info(f"加载模型: {model_path}")
    logger.info(f"编码器配置: encoder_type={encoder_type}, hidden_dim={hidden_dim}, "
                f"n_heads={n_heads}, n_attn_layers={n_attn_layers}")

    features_extractor_kwargs = dict(
        hidden_dim=hidden_dim,
        encoder_type=encoder_type,
        n_heads=n_heads,
        n_attn_layers=n_attn_layers,
    )

    model = PPO.load(
        model_path,
        custom_objects={
            "IncrementalLevelBundleFeaturesExtractor": IncrementalLevelBundleFeaturesExtractor,
            "IncrementalLevelBundleActorCriticPolicy": IncrementalLevelBundleActorCriticPolicy,
            "policy_kwargs": {
                "features_extractor_class": IncrementalLevelBundleFeaturesExtractor,
                "features_extractor_kwargs": features_extractor_kwargs,
                "net_arch": dict(pi=[hidden_dim, hidden_dim], vf=[hidden_dim, hidden_dim]),
            },
        }
    )
    return model


# ============================================================
# JSON 保存
# ============================================================

def save_results_to_json(all_results, save_dir):
    """将所有结果保存为 JSON"""
    results_file = os.path.join(save_dir, 'all_results.json')

    save_data = []
    for result in all_results:
        data = {
            "config_info": result["config_info"],
            "baseline": {
                "final_lb": float(result["baseline"].get("final_lb", 0)),
                "final_ub": float(result["baseline"].get("final_ub", 0)),
                "total_time": float(result["baseline"].get("total_time", 0)),
                "time_history": [float(x) for x in result["baseline"].get("time", [])],
                "gap": [float(x) for x in result["baseline"].get("gap", [])],
            },
            "rl": {
                "lb_history": [float(x) for x in result["rl"].get("lb_history", [])],
                "ub_history": [float(x) for x in result["rl"].get("ub_history", [])],
                "reward_history": [float(x) for x in result["rl"].get("reward_history", [])],
                "time_history": [float(x) for x in result["rl"].get("time", [])],
                "gap": [float(x) for x in result["rl"].get("gap", [])],
                "switch_step": result["rl"].get("switch_step"),
            },
            "rl_warmstart": {
                "lb_history": [float(x) for x in result["rl_warmstart"].get("lb_history", [])],
                "ub_history": [float(x) for x in result["rl_warmstart"].get("ub_history", [])],
                "reward_history": [float(x) for x in result["rl_warmstart"].get("reward_history", [])],
                "time_history": [float(x) for x in result["rl_warmstart"].get("time", [])],
                "gap": [float(x) for x in result["rl_warmstart"].get("gap", [])],
                "switch_step": result["rl_warmstart"].get("switch_step"),
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_data.append(data)

    with open(results_file, 'w', encoding='utf-8') as f:
        for data in save_data:
            json.dump(data, f)
            f.write('\n')

    print(f"Results saved to {results_file}")


# ============================================================
# 主测试流程
# ============================================================

def run_test_for_configs(configs, config_info_list, experiment_name, logger, model,
                         K=20, warmstart_threshold=1e-4, patience=3):
    """对一组 config 运行测试"""
    all_results = []

    for idx, (config, config_info) in enumerate(zip(configs, config_info_list)):
        logger.info(f"=== Testing config {idx+1}/{len(configs)}: "
                     f"i={config_info['i']}, t={config_info['t']}, n={config_info['n']} ===")

        # 1. Baseline
        logger.info("Running Baseline...")
        baseline_lb, baseline_ub, baseline_time, baseline_lb_hist, baseline_ub_hist, baseline_time_hist = level_bundle_baseline(
            logger, config, n=config_info['n']
        )
        logger.info(f"Baseline: LB={baseline_lb:.6f}, UB={baseline_ub:.6f}, time={baseline_time:.4f}s")
        baseline_gap = compute_opt_gap(baseline_lb_hist, baseline_ub_hist)

        # 2. RL
        logger.info("Running RL...")
        rl_env = IncrementalLevelBundleEnv.create_env(logger, config, K=K, verbose=True, use_outer=True)
        rl_lb, rl_reward, rl_time, rl_ub, _ = level_bundle_rl(
            rl_env, model, logger, deterministic=True, K=K
        )
        rl_gap = compute_opt_gap(rl_lb, rl_ub)
        rl_env.close()

        # 3. RL Warmstart
        logger.info("Running RL Warmstart...")
        ws_env = IncrementalLevelBundleEnv.create_env(logger, config, K=K, verbose=True, use_outer=True)
        ws_lb, ws_reward, ws_time, ws_ub, _, ws_switch_step = level_bundle_rl_warmstart(
            ws_env, model, logger, deterministic=True, K=K,
            warmstart_threshold=warmstart_threshold, patience=patience,
        )
        ws_gap = compute_opt_gap(ws_lb, ws_ub)
        ws_env.close()

        # 4. 保存结果
        result = {
            "config_info": config_info,
            "baseline": {
                "final_lb": baseline_lb,
                "final_ub": baseline_ub,
                "total_time": baseline_time,
                "time": baseline_time_hist,
                "ub_history": baseline_ub_hist,
                "gap": baseline_gap,
            },
            "rl": {
                "lb_history": rl_lb,
                "ub_history": rl_ub,
                "reward_history": rl_reward,
                "time": rl_time,
                "gap": rl_gap,
                "switch_step": None,
            },
            "rl_warmstart": {
                "lb_history": ws_lb,
                "ub_history": ws_ub,
                "reward_history": ws_reward,
                "time": ws_time,
                "gap": ws_gap,
                "switch_step": ws_switch_step,
            },
        }

        all_results.append(result)
        logger.info(f"=== Finished config {idx+1}/{len(configs)} ===")

    return all_results


def main(experiment_name, train_experiment_name=None, i=1, t=5, K=20,
         hidden_dim=128, encoder_type="self_attention", n_heads=4, n_attn_layers=2,
         auto_load_config=True, warmstart_threshold=1e-4, patience=3):
    """主测试函数"""
    if train_experiment_name is None:
        train_experiment_name = experiment_name

    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    base_dir = str(base_dir)

    save_dir = os.path.join(base_dir, "test_result", experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    logger = get_logger(os.path.join(save_dir, "test.log"))

    # 自动从训练配置加载网络参数
    if auto_load_config:
        train_config = load_train_config(train_experiment_name)
        if train_config is not None:
            net_cfg = train_config.get('network', {})
            if 'hidden_dim' in net_cfg:
                hidden_dim = net_cfg['hidden_dim']
            if 'encoder_type' in net_cfg:
                encoder_type = net_cfg['encoder_type']
            if 'n_heads' in net_cfg:
                n_heads = net_cfg['n_heads']
            if 'n_attn_layers' in net_cfg:
                n_attn_layers = net_cfg['n_attn_layers']
            logger.info(f"从训练配置加载网络参数: hidden_dim={hidden_dim}, "
                        f"encoder_type={encoder_type}, n_heads={n_heads}, n_attn_layers={n_attn_layers}")

    # 加载模型
    logger.info(f"加载模型: {train_experiment_name}")
    model = load_latest_model(
        train_experiment_name, logger,
        hidden_dim=hidden_dim,
        encoder_type=encoder_type,
        n_heads=n_heads,
        n_attn_layers=n_attn_layers,
    )

    # 收集 configs
    logger.info(f"Collecting configs ...")
    configs, config_info_list = collect_configs()
    logger.info(f"Loaded {len(configs)} configs")

    if not configs:
        logger.error("No configs found!")
        return

    # 运行测试
    logger.info("Running tests for all configs...")
    all_results = run_test_for_configs(
        configs, config_info_list, experiment_name, logger, model,
        K=K, warmstart_threshold=warmstart_threshold, patience=patience,
    )

    # 计算平均结果
    logger.info("Computing average results...")
    avg_results = compute_average_results(all_results)

    # 保存结果
    logger.info("Saving results...")
    save_results_to_json(all_results, save_dir)

    # 绘制图表
    logger.info("Plotting results...")
    plot_results(avg_results, save_dir, experiment_name=experiment_name)

    logger.info("All tests completed!")


# ============================================================
# Config 收集
# ============================================================

def collect_configs():
    """收集指定 i, t 的所有 config"""
    configs = []
    config_info = []

    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    config_dir = Path(os.path.join(base_dir, "configs"))

    i = 2
    for t in range(1, 5):
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


if __name__ == "__main__":
    experiment_name = "inc_exp_01"
    main(
        experiment_name=experiment_name,
        train_experiment_name=experiment_name,
        K=20,
    )
