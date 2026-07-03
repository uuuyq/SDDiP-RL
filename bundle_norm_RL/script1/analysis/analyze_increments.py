"""
增量统计分析脚本

加载所有 config，执行增量形式的 Level Bundle 迭代，
收集每一步的增量 d = (d_π, d_{π0}) 数据，
统计增量的大小范围、分布特征，为 RL 预测增量做准备。
"""

import os
import sys
import glob
import logging

# 确保项目根目录在 sys.path 中
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
from time import time

from bundle_norm_RL.script.config import LevelBundleConfig
from bundle_norm_RL.script.logger import get_logger
from bundle_norm_RL.script1.level_bundle_problem import (
    LevelBundleSolver,
    InnerProblem,
    IncrementalOuterProblem,
    SolverResults,
)


def run_incremental_bundle(config, n, logger, rho=1.0):
    """
    执行增量形式 Level Bundle 迭代，收集增量历史

    Returns:
        dict: 包含增量历史和求解结果
    """
    X_trial = config.X_trial
    theta_trial = config.THETA_TRIAL

    # 初始化 inner problem
    inner_problem = InnerProblem(logger, config, n=n)

    # 初始化增量 outer problem
    outer_problem = IncrementalOuterProblem(
        logger,
        dim_pi=config.N_VARS,
        X_trial=config.X_trial,
        theta_trial=config.THETA_TRIAL,
        rho=rho,
    )

    # 初始化
    pi_hat = np.zeros(len(X_trial))
    pi0_hat = 0.1
    pi_bar = np.zeros(len(X_trial))
    pi0_bar = 0.1

    pi_star = None
    pi0_star = None
    LB = float('-inf')
    UB = float('inf')

    # 收集增量历史
    d_pi_history = []       # 每步的 d_π (list of arrays)
    d_pi0_history = []      # 每步的 d_{π0}
    d_norm_history = []     # 每步的 ||d||
    pi_bar_history = []     # 每步更新后的稳定中心 π̄
    pi0_bar_history = []    # 每步更新后的稳定中心 π̄₀
    lambda_history = []     # 每步的 λ = π̄ + d
    lb_history = []
    ub_history = []
    serious_step_indices = []  # serious step 的迭代索引

    converged = False
    n_iterations = 0

    for iter_idx in range(config.iteration_limit):
        n_iterations = iter_idx + 1

        # Step 1: 求解 inner model
        z_X_values, obj_term_value, inner_obj = inner_problem.solve(pi_hat, pi0_hat)
        if z_X_values is None:
            break

        subgradient = z_X_values + [obj_term_value]
        lambda_i = np.concatenate([pi_hat, [pi0_hat]])
        omega_i = inner_obj - pi_hat @ X_trial - pi0_hat * theta_trial

        # Step 2: 添加 cut
        outer_problem.add_cut(subgradient, lambda_i, omega_i)

        # Step 3: 求解 outer model (最大化模式)
        pi_dummy, pi0_dummy, outer_obj = outer_problem.solve()
        if outer_obj is None:
            break

        # Step 4: 更新 LB/UB
        gap = inner_obj - pi_hat @ X_trial - pi0_hat * theta_trial
        if gap > LB:
            LB = gap
            pi_star = pi_hat.copy()
            pi0_star = pi0_hat
        UB = outer_obj

        lb_history.append(LB)
        ub_history.append(UB)

        # Step 5: 判断收敛
        if UB - LB < config.gap_tol * abs(UB) or UB - LB < 1e-6:
            if pi0_star > 1e-6 and LB / pi0_star >= config.pi0_tol * (abs(theta_trial) + 1):
                converged = True
            break

        # Step 6: Level 策略
        level = UB - config.level_factor * (UB - LB)
        outer_problem.set_level(level)
        outer_problem.model.params.Method = 2
        outer_problem.model.update()
        outer_problem.model.optimize()

        if outer_problem.model.status != 2:
            for method in [1, 0]:
                outer_problem.model.params.Method = method
                outer_problem.model.update()
                outer_problem.model.optimize()
                if outer_problem.model.status == 2:
                    break

        if outer_problem.model.status != 2:
            outer_problem.recover()
            pi_hat = np.array(pi_dummy)
            pi0_hat = pi0_dummy
            continue

        # 恢复乘子: λ = λ̄ + d*
        d_pi_star = np.array([outer_problem.d_pi[i].x for i in range(config.N_VARS)])
        d_pi0_star = outer_problem.d_pi0.x

        # 记录增量
        d_pi_history.append(d_pi_star.copy())
        d_pi0_history.append(d_pi0_star)
        d_norm_history.append(np.linalg.norm(np.concatenate([d_pi_star, [d_pi0_star]])))

        pi_hat = outer_problem.pi_bar + d_pi_star
        pi0_hat = outer_problem.pi0_bar + d_pi0_star
        lambda_history.append(np.concatenate([pi_hat, [pi0_hat]]))

        outer_problem.recover()

        # Serious step
        current_dual = inner_obj - pi_hat @ X_trial - pi0_hat * theta_trial
        if current_dual > LB + 1e-8 * max(1.0, abs(LB)):
            pi_bar = pi_hat.copy()
            pi0_bar = pi0_hat
            outer_problem.set_stability_center(pi_bar, pi0_bar)
            pi_bar_history.append(pi_bar.copy())
            pi0_bar_history.append(pi0_bar)
            serious_step_indices.append(iter_idx)

    return {
        'converged': converged,
        'n_iterations': n_iterations,
        'LB': LB,
        'UB': UB,
        'd_pi_history': d_pi_history,
        'd_pi0_history': d_pi0_history,
        'd_norm_history': d_norm_history,
        'pi_bar_history': pi_bar_history,
        'pi0_bar_history': pi0_bar_history,
        'lambda_history': lambda_history,
        'lb_history': lb_history,
        'ub_history': ub_history,
        'serious_step_indices': serious_step_indices,
    }


def analyze_increments(all_results):
    """统计分析所有增量数据"""
    # 汇总所有增量
    all_d_pi = []
    all_d_pi0 = []
    all_d_norm = []
    all_d_pi_by_dim = None  # 按维度分组

    n_converged = 0
    n_total = len(all_results)
    total_iters = 0

    for r in all_results:
        if r is None:
            continue
        if r['converged']:
            n_converged += 1
        total_iters += r['n_iterations']

        d_pi_list = r['d_pi_history']
        d_pi0_list = r['d_pi0_history']
        d_norm_list = r['d_norm_history']

        if not d_pi_list:
            continue

        all_d_norm.extend(d_norm_list)
        all_d_pi0.extend(d_pi0_list)

        for d_pi in d_pi_list:
            all_d_pi.append(d_pi)
            if all_d_pi_by_dim is None:
                all_d_pi_by_dim = [[] for _ in range(len(d_pi))]
            for j in range(len(d_pi)):
                all_d_pi_by_dim[j].append(d_pi[j])

    print("=" * 70)
    print("增量统计分析报告")
    print("=" * 70)
    print(f"\n总案例数: {n_total}")
    print(f"收敛案例数: {n_converged} ({n_converged/max(n_total,1)*100:.1f}%)")
    print(f"总迭代次数: {total_iters}")
    print(f"平均迭代次数: {total_iters/max(n_total,1):.1f}")
    print(f"总增量记录数: {len(all_d_norm)}")

    if not all_d_norm:
        print("\n无增量数据可分析")
        return

    # === 整体增量范数统计 ===
    d_norm_arr = np.array(all_d_norm)
    print(f"\n{'='*70}")
    print("1. increment norm ||d|| = ||(d_pi, d_pi0)|| statistics")
    print(f"{'='*70}")
    print(f"  最小值:   {d_norm_arr.min():.6e}")
    print(f"  最大值:   {d_norm_arr.max():.6e}")
    print(f"  均值:     {d_norm_arr.mean():.6e}")
    print(f"  中位数:   {np.median(d_norm_arr):.6e}")
    print(f"  标准差:   {d_norm_arr.std():.6e}")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  P{p:02d}:      {np.percentile(d_norm_arr, p):.6e}")

    # === d_{π0} 统计 ===
    d_pi0_arr = np.array(all_d_pi0)
    print(f"\n{'='*70}")
    print("2. d_pi0 (pi0 的增量) 统计")
    print(f"{'='*70}")
    print(f"  最小值:   {d_pi0_arr.min():.6e}")
    print(f"  最大值:   {d_pi0_arr.max():.6e}")
    print(f"  均值:     {d_pi0_arr.mean():.6e}")
    print(f"  中位数:   {np.median(d_pi0_arr):.6e}")
    print(f"  标准差:   {d_pi0_arr.std():.6e}")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  P{p:02d}:      {np.percentile(d_pi0_arr, p):.6e}")

    # === d_π 各维度统计 ===
    if all_d_pi_by_dim is not None:
        print(f"\n{'='*70}")
        print("3. d_pi per-dimension statistics")
        print(f"{'='*70}")
        print(f"  dim: {len(all_d_pi_by_dim)}")
        for j in range(len(all_d_pi_by_dim)):
            arr = np.array(all_d_pi_by_dim[j])
            print(f"  d_pi[{j:2d}]: min={arr.min():+.4e}, max={arr.max():+.4e}, "
                  f"mean={arr.mean():+.4e}, std={arr.std():.4e}, "
                  f"|max|={np.abs(arr).max():.4e}")

    # === d_π 各维度绝对值汇总 ===
    if all_d_pi_by_dim is not None:
        print(f"\n{'='*70}")
        print("4. |d_pi| per-dimension statistics (absolute value)")
        print(f"{'='*70}")
        all_abs_d_pi = []
        for j in range(len(all_d_pi_by_dim)):
            abs_arr = np.abs(np.array(all_d_pi_by_dim[j]))
            all_abs_d_pi.append(abs_arr)
            print(f"  |d_pi[{j:2d}]|: min={abs_arr.min():.4e}, max={abs_arr.max():.4e}, "
                  f"mean={abs_arr.mean():.4e}, P95={np.percentile(abs_arr, 95):.4e}")

        all_abs_flat = np.concatenate(all_abs_d_pi)
        print(f"\n  All |d_pi| summary:")
        print(f"    最小值:   {all_abs_flat.min():.6e}")
        print(f"    最大值:   {all_abs_flat.max():.6e}")
        print(f"    均值:     {all_abs_flat.mean():.6e}")
        print(f"    中位数:   {np.median(all_abs_flat):.6e}")
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            print(f"    P{p:02d}:      {np.percentile(all_abs_flat, p):.6e}")

    # === 按 |d| 分段统计 ===
    print(f"\n{'='*70}")
    print("5. ||d|| 分段统计")
    print(f"{'='*70}")
    bins = [0, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, float('inf')]
    bin_labels = ['[0, 1e-4)', '[1e-4, 1e-3)', '[1e-3, 1e-2)', '[1e-2, 1e-1)',
                  '[1e-1, 0.5)', '[0.5, 1.0)', '[1.0, 5.0)', '[5.0, +∞)']
    for i in range(len(bins) - 1):
        count = np.sum((d_norm_arr >= bins[i]) & (d_norm_arr < bins[i+1]))
        pct = count / len(d_norm_arr) * 100
        print(f"  {bin_labels[i]:18s}: {count:5d} ({pct:5.1f}%)")

    # === 迭代过程中增量的变化趋势 ===
    print(f"\n{'='*70}")
    print("6. 迭代过程中 ||d|| 的变化趋势 (按相对迭代位置分组)")
    print(f"{'='*70}")
    # 收集每个案例中增量的归一化迭代位置
    early_norms = []   # 前 1/3 迭代
    mid_norms = []     # 中间 1/3 迭代
    late_norms = []    # 后 1/3 迭代
    for r in all_results:
        if r is None or not r['d_norm_history']:
            continue
        n = len(r['d_norm_history'])
        norms = r['d_norm_history']
        t1 = max(1, n // 3)
        t2 = max(1, 2 * n // 3)
        early_norms.extend(norms[:t1])
        mid_norms.extend(norms[t1:t2])
        late_norms.extend(norms[t2:])

    for label, norms in [("前1/3迭代", early_norms), ("中1/3迭代", mid_norms), ("后1/3迭代", late_norms)]:
        if norms:
            arr = np.array(norms)
            print(f"  {label}: mean={arr.mean():.4e}, std={arr.std():.4e}, "
                  f"median={np.median(arr):.4e}, max={arr.max():.4e}")

    return {
        'd_norm_arr': d_norm_arr,
        'd_pi0_arr': d_pi0_arr,
        'all_d_pi': all_d_pi,
        'all_d_pi_by_dim': all_d_pi_by_dim,
    }


def main():
    configs_dir = r"D:\tools\workspace_pycharm\SDDiP-RL\bundle_norm_RL\configs"
    # 日志和数据输出到当前 analysis 目录
    analysis_dir = os.path.dirname(os.path.abspath(__file__))
    logger = get_logger(os.path.join(analysis_dir, "increment_analysis.log"))

    # 收集所有 config 文件
    pkl_files = sorted(glob.glob(os.path.join(configs_dir, "config_*.pkl")))
    logger.info(f"Found {len(pkl_files)} config files")

    all_results = []
    failed = 0

    for idx, pkl_path in enumerate(pkl_files):
        # 从文件名解析参数
        basename = os.path.basename(pkl_path)
        # config_{iteration}_{stage}_{realization}.pkl
        parts = basename.replace("config_", "").replace(".pkl", "").split("_")
        iteration, t, n = int(parts[0]), int(parts[1]), int(parts[2])

        try:
            config = LevelBundleConfig.from_pkl(pkl_path)
            logger.info(f"[{idx+1}/{len(pkl_files)}] Processing {basename} (i={iteration}, t={t}, n={n})")

            result = run_incremental_bundle(config, n=n, logger=logger, rho=1.0)
            all_results.append(result)

            status = "converged" if result['converged'] else "NOT converged"
            logger.info(f"  -> {status}, iters={result['n_iterations']}, "
                        f"LB={result['LB']:.4f}, UB={result['UB']:.4f}, "
                        f"d_norms={len(result['d_norm_history'])}")

        except Exception as e:
            logger.warning(f"  -> FAILED: {e}")
            all_results.append(None)
            failed += 1

    logger.info(f"\nCompleted: {len(pkl_files)} configs, {failed} failed")

    # 统计分析
    analysis = analyze_increments(all_results)

    # 保存原始数据
    output_dir = analysis_dir
    np.savez(
        os.path.join(output_dir, "increment_stats.npz"),
        d_norm=analysis['d_norm_arr'] if analysis else [],
        d_pi0=analysis['d_pi0_arr'] if analysis else [],
    )
    logger.info(f"\nData saved to {os.path.join(output_dir, 'increment_stats.npz')}")


if __name__ == "__main__":
    main()
