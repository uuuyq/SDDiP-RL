"""
benchmark.py

对比两种求解方式：
1. warmstart: 使用 fast_cut_gen 生成的 cuts 作为初始 cuts，然后继续 bundle 算法求解
2. baseline: 传统 bundle 算法从零开始求解

记录并对比：
- 上下界变化
- 迭代次数
- 总耗时
"""

import json
import time

import numpy as np
import matplotlib.pyplot as plt

from bundle_fast.lag_problem import SubProblem, MasterProblem
from bundle_fast.logger import get_logger
from bundle_fast import config
from bundle_fast import fast_g_gen
from bundle_fast import fast_cut_gen


def run_bundle_warmstart(
    max_iterations: int = 1000,
    tolerance: float = 1e-5,
    step: int = 1,
    solution_size: int = 10,
    realization: int = 1,
    logger=None,
):
    """
    使用 warmstart cuts 的 bundle 方法

    Args:
        max_iterations: 最大迭代次数
        tolerance: 收敛阈值
        step: 每次增加的 subgradient 数量
        solution_size: 筛选的 top solution 数量
        realization: 场景索引
        logger: 日志器

    Returns:
        result: 包含收敛历史的字典
    """
    if logger is None:
        logger = get_logger("log/benchmark_warmstart.log")

    trial_point = (config.X_TRIAL, config.Y_TRIAL, config.X_BS_TRIAL, config.SOC_TRIAL)
    problem_params = config.PROBLEM_PARAMS
    t = config.T

    # ========== 第一步：生成 cuts（调用 fast_g_gen 和 fast_cut_gen）==========

    logger.info(">>> 步骤1: 历史解收集 (fast_g_gen.history_solution_collect) <<<")
    mu_weights, solution_collection = fast_g_gen.history_solution_collect(realization=0)

    cut_gen_start_time = time.time()

    logger.info(">>> 步骤2: 生成 subgradients (fast_g_gen.bundle_fast) <<<")
    subgradients, mu_list = fast_g_gen.bundle_fast(mu_weights, solution_collection, solution_size, realization=realization)

    logger.info(">>> 步骤3: 生成 cuts (fast_cut_gen.generate_cuts) <<<")
    cuts = fast_cut_gen.generate_cuts(
        subgradients=subgradients,
        mu_weights=mu_list,
        step=step,
        t=t,
        realization=realization,
        trial_point=trial_point,
        problem_params=problem_params,
        logger=logger,
    )

    cut_gen_time = time.time() - cut_gen_start_time
    logger.info(f"Cuts 生成耗时: {cut_gen_time:.3f}s, 共 {len(cuts)} 个 cuts")

    # ========== 第二步：使用 cuts 进行 bundle 迭代 ==========
    bundle_start_time = time.time()

    # 初始化子问题和主问题
    sub = SubProblem(logger, problem_params, trial_point, t, realization, 0)
    master = MasterProblem(logger, config.N_VARS, tolerance=tolerance)

    # 将预生成的 cuts 添加到主问题
    for cut in cuts:
        g = np.array(cut["g"])
        x = np.array(cut["x"])
        f = cut["f"]
        master.add_cut(x, f, g)

    # 记录历史
    ub_history = []
    lb_history = []
    iter_times = []

    # 第一次：用主问题求解，得到初始 x 和 ub
    ub, x_new = master.solve_master()
    g_new, f_new = sub.solve(x_new)

    # 初始化 f_best 和 x_best
    master.f_best = f_new
    master.x_best = x_new.copy()

    ub_history.append(ub)
    lb_history.append(f_new)
    iter_times.append(time.time() - bundle_start_time)

    logger.info(f"Warmstart 初始: ub={ub:.6f}, lb={f_new:.6f}, gap={ub - f_new:.6f}")

    # 继续迭代
    for i in range(max_iterations):
        iter_start = time.time()

        master.add_cut(x_new, f_new, g_new)
        ub, x_new = master.solve_master()
        g_new, f_new = sub.solve(x_new)
        serious_step, delta, stop_flag = master.update_strategy(x_new, f_new, g_new, ub)

        iter_time = time.time() - iter_start
        iter_times.append(time.time() - bundle_start_time)

        ub_history.append(ub)
        lb_history.append(master.f_best)

        logger.info(f"迭代 {i+1}: ub={ub:.6f}, lb={master.f_best:.6f}, delta={delta:.6f}, time={iter_time:.3f}s")

        if stop_flag:
            logger.info(f"Warmstart 收敛于迭代 {i+1}")
            break

    bundle_time = time.time() - bundle_start_time
    total_time = cut_gen_time + bundle_time

    result = {
        "method": "warmstart",
        "n_initial_cuts": len(cuts),
        "cut_gen_time": cut_gen_time,
        "bundle_time": bundle_time,
        "total_time": total_time,
        "iterations": len(ub_history),
        "ub_history": ub_history,
        "lb_history": lb_history,
        "iter_times": iter_times,
        "final_ub": ub_history[-1],
        "final_lb": lb_history[-1],
        "final_gap": ub_history[-1] - lb_history[-1],
    }

    return result


def run_bundle_baseline(
    max_iterations: int = 1000,
    tolerance: float = 1e-5,
    realization: int = 1,
    logger=None,
):
    """
    传统 bundle 方法（从零开始）

    Args:
        max_iterations: 最大迭代次数
        tolerance: 收敛阈值
        realization: 场景索引
        logger: 日志器

    Returns:
        result: 包含收敛历史的字典
    """
    if logger is None:
        logger = get_logger("log/benchmark_baseline.log")

    trial_point = (config.X_TRIAL, config.Y_TRIAL, config.X_BS_TRIAL, config.SOC_TRIAL)
    problem_params = config.PROBLEM_PARAMS
    t = config.T

    # 初始化子问题和主问题
    sub = SubProblem(logger, problem_params, trial_point, t, realization, 0)
    master = MasterProblem(logger, config.N_VARS, tolerance=tolerance)

    # 记录历史
    ub_history = []
    lb_history = []
    iter_times = []

    start_time = time.time()

    # 初始化：从 x=0 开始求解子问题（只有下界，没有上界）
    x_new = np.zeros(config.N_VARS)
    g_new, f_new = sub.solve(x_new)
    master.update_strategy(x_new, f_new, g_new, ub=None)

    # 记录初始点：下界为子问题求解值，上界设为一个很大的值
    ub_history.append(1e4)
    lb_history.append(f_new)
    iter_times.append(time.time() - start_time)
    logger.info(f"Baseline 初始子问题求解: lb={f_new:.6f}, ub=1e4 (初始值)")

    # 继续迭代
    for i in range(max_iterations):
        iter_start = time.time()

        master.add_cut(x_new, f_new, g_new)
        ub, x_new = master.solve_master()
        g_new, f_new = sub.solve(x_new)
        serious_step, delta, stop_flag = master.update_strategy(x_new, f_new, g_new, ub)

        iter_time = time.time() - iter_start
        iter_times.append(time.time() - start_time)

        ub_history.append(ub)
        lb_history.append(master.f_best)

        logger.info(f"迭代 {i+1}: ub={ub:.6f}, lb={master.f_best:.6f}, delta={delta:.6f}, time={iter_time:.3f}s")

        if stop_flag:
            logger.info(f"Baseline 收敛于迭代 {i+1}")
            break

    total_time = time.time() - start_time

    result = {
        "method": "baseline",
        "iterations": len(ub_history),
        "total_time": total_time,
        "ub_history": ub_history,
        "lb_history": lb_history,
        "iter_times": iter_times,
        "final_ub": ub_history[-1],
        "final_lb": lb_history[-1],
        "final_gap": ub_history[-1] - lb_history[-1],
    }

    return result


def plot_convergence(result: dict, output_file: str):
    """绘制收敛曲线"""
    plt.figure(figsize=(10, 6))

    iterations = range(1, len(result["ub_history"]) + 1)

    plt.plot(iterations, result["ub_history"], 'b-o', label='Upper Bound', markersize=3)
    plt.plot(iterations, result["lb_history"], 'r-s', label='Lower Bound', markersize=3)

    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title(f'Convergence - {result["method"]}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"已保存收敛曲线到 {output_file}")


def save_result(result: dict, output_file: str):
    """保存结果到 JSON"""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"已保存结果到 {output_file}")


def main(
    max_iterations: int = 1000,
    tolerance: float = 1e-5,
    step: int = 1,
    solution_size: int = 10,
    realization: int = 1,
):
    """
    主函数：运行两种方法的对比

    Args:
        max_iterations: 最大迭代次数
        tolerance: 收敛阈值
        step: 每次增加的 subgradient 数量
        solution_size: 筛选的 top solution 数量
        realization: 场景索引
    """
    logger = get_logger("log/benchmark.log")
    logger.info("=" * 50)
    logger.info("开始 Benchmark 对比测试")
    logger.info(f"参数: t={config.T}, max_iter={max_iterations}, tol={tolerance}, step={step}, solution_size={solution_size}, realization={realization}")
    logger.info("=" * 50)

    # 运行 warmstart 方法
    logger.info("\n>>> 运行 Warmstart 方法 <<<")
    result_warmstart = run_bundle_warmstart(
        max_iterations=max_iterations,
        tolerance=tolerance,
        step=step,
        solution_size=solution_size,
        realization=realization,
        logger=logger,
    )

    # 运行 baseline 方法
    logger.info("\n>>> 运行 Baseline 方法 <<<")
    result_baseline = run_bundle_baseline(
        max_iterations=max_iterations,
        tolerance=tolerance,
        realization=realization,
        logger=logger,
    )

    # 保存结果
    save_result(result_warmstart, f"benchmark_warmstart_{config.T}_{realization}.json")
    save_result(result_baseline, f"benchmark_baseline_{config.T}_{realization}.json")

    # 绘制收敛曲线
    plot_convergence(result_warmstart, f"benchmark_warmstart_{config.T}_{realization}.png")
    plot_convergence(result_baseline, f"benchmark_baseline_{config.T}_{realization}.png")

    # 打印对比摘要
    print("\n" + "=" * 60)
    print("对比摘要")
    print("=" * 60)
    print(f"{'指标':<25} {'Warmstart':<20} {'Baseline':<15}")
    print("-" * 60)
    print(f"{'迭代次数':<25} {result_warmstart['iterations']:<20} {result_baseline['iterations']:<15}")
    print(f"{'Cuts生成耗时 (s)':<25} {result_warmstart['cut_gen_time']:.3f}{'':<16} {'-':<15}")
    print(f"{'Bundle迭代耗时 (s)':<25} {result_warmstart['bundle_time']:.3f}{'':<16} {result_baseline['total_time']:.3f}")
    print(f"{'总耗时 (s)':<25} {result_warmstart['total_time']:.3f}{'':<16} {result_baseline['total_time']:.3f}")
    print(f"{'最终上界':<25} {result_warmstart['final_ub']:.6f}{'':<9} {result_baseline['final_ub']:.6f}")
    print(f"{'最终下界':<25} {result_warmstart['final_lb']:.6f}{'':<9} {result_baseline['final_lb']:.6f}")
    print(f"{'最终 Gap':<25} {result_warmstart['final_gap']:.6f}{'':<9} {result_baseline['final_gap']:.6f}")
    print("=" * 60)

    return result_warmstart, result_baseline


if __name__ == "__main__":
    main(
        max_iterations=1000,
        tolerance=1e-5,
        step=1,
        solution_size=10,
        realization=1,
    )