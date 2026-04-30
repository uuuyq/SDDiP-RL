"""
bundle_fast_script.py

完整的 bundle_fast 流程：
1. fast_g_gen.bundle_fast - 生成 subgradients
2. fast_cut_gen.generate_cuts - 使用 subgradients 生成 cuts
3. run_bundle_with_cuts - 使用 cuts 进行 bundle 迭代直到收敛

输出：
- f_best, x_best, iterations, solver_time
"""

import time

import numpy as np

from bundle_fast.lag_problem import SubProblem, MasterProblem
from bundle_fast.logger import get_logger
from bundle_fast import fast_g_gen
from bundle_fast import fast_cut_gen
from bundle_fast.config import BundleConfig


def generate_cuts(
    config: BundleConfig,
    subgradients,
    mu_weights,
    step: int,
    realization: int = 1,
    logger=None,
):
    """
    使用 subgradients 生成 cuts

    Args:
        config: BundleConfig 配置对象
        subgradients: 梯度列表
        mu_weights: 权重列表
        step: 每次增加的梯度数量
        realization: 场景索引
        logger: 日志器

    Returns:
        cuts: cut 列表
    """
    if logger is None:
        logger = get_logger("log/bundle_fast_script.log")

    cuts = fast_cut_gen.generate_cuts(
        config=config,
        subgradients=subgradients,
        mu_weights=mu_weights,
        step=step,
        realization=realization,
        logger=logger,
    )

    return cuts


def run_bundle_with_cuts(
    config: BundleConfig,
    cuts,
    max_iterations: int = 1000,
    tolerance: float = 1e-5,
    realization: int = 1,
    logger=None,
):
    """
    使用 cuts 进行 bundle 迭代直到收敛

    Args:
        config: BundleConfig 配置对象
        cuts: cut 列表
        max_iterations: 最大迭代次数
        tolerance: 收敛阈值
        realization: 场景索引
        logger: 日志器

    Returns:
        f_best, x_best, iterations, solver_time
    """
    if logger is None:
        logger = get_logger("log/bundle_fast_script.log")

    trial_point = config.trial_point

    # 初始化子问题和主问题
    sub = SubProblem(
        logger, config.PROBLEM_PARAMS, trial_point, config.T, realization, 0
    )
    master = MasterProblem(logger, config.N_VARS, tolerance=tolerance)

    # 将预生成的 cuts 添加到主问题
    for cut in cuts:
        g = np.array(cut["g"])
        x = np.array(cut["x"])
        f = cut["f"]
        master.add_cut(x, f, g)

    logger.info(f"加载了 {len(cuts)} 个 warmstart cuts")

    # 第一次：用主问题求解，得到初始 x 和 ub
    ub, x_new = master.solve_master()
    g_new, f_new = sub.solve(x_new)

    # 初始化 f_best 和 x_best
    master.f_best = f_new
    master.x_best = x_new.copy()

    logger.info(f"初始: ub={ub:.6f}, lb={f_new:.6f}, gap={ub - f_new:.6f}")

    # 迭代直到收敛
    for i in range(max_iterations):
        master.add_cut(x_new, f_new, g_new)
        ub, x_new = master.solve_master()
        g_new, f_new = sub.solve(x_new)
        serious_step, delta, stop_flag = master.update_strategy(x_new, f_new, g_new, ub)

        logger.info(
            f"迭代 {i+1}: ub={ub:.6f}, lb={master.f_best:.6f}, delta={delta:.6f}"
        )

        if stop_flag:
            logger.info(f"收敛于迭代 {i+1}")
            break

    return master.f_best, master.x_best.copy(), i + 1


def run(
    config: BundleConfig,
    mu_weights,
    solution_collection,
    size,
    step: int = 1,
    max_iterations: int = 1000,
    tolerance: float = 1e-5,
    realization: int = 1,
    logger=None,
):
    """
    完整流程：
    1. fast_g_gen.history_solution_collect - 收集历史解和 mu 权重
    2. fast_g_gen.bundle_fast - 生成 subgradients
    3. fast_cut_gen.generate_cuts - 使用 subgradients 生成 cuts
    4. run_bundle_with_cuts - 使用 cuts 进行 bundle 迭代直到收敛

    Args:
        config: BundleConfig 配置对象
        mu_weights: 权重列表
        solution_collection: 历史解集合
        size: 筛选的 top solution 数量
        step: 每次增加的梯度数量
        max_iterations: 最大迭代次数
        tolerance: 收敛阈值
        realization: 场景索引
        logger: 日志器

    Returns:
        f_best, x_best, iterations, solver_time
    """
    if logger is None:
        logger = get_logger("log/bundle_fast_script.log")

    start_time = time.time()

    logger.info("=" * 40)
    logger.info("步骤1: 生成 subgradients (fast_g_gen.bundle_fast)")
    subgradients, mu_list, z_vars_list, x_vars_list = fast_g_gen.gen_subgradient(
        config=config,
        mu_weights=mu_weights,
        solution_collection=solution_collection,
        size=size,
        realization=1,
        logger=logger,
    )

    logger.info("=" * 40)
    logger.info("步骤2: 生成 cuts (fast_cut_gen.generate_cuts)")
    cuts = generate_cuts(
        config=config,
        subgradients=subgradients,
        mu_weights=mu_list,
        step=step,
        realization=realization,
        logger=logger,
    )

    logger.info("=" * 40)
    logger.info("步骤3: Bundle 迭代直到收敛 (run_bundle_with_cuts)")
    f_best, x_best, iterations = run_bundle_with_cuts(
        config=config,
        cuts=cuts,
        max_iterations=max_iterations,
        tolerance=tolerance,
        realization=realization,
        logger=logger,
    )

    solver_time = time.time() - start_time
    logger.info("=" * 40)
    logger.info(f"完成! 总耗时: {solver_time:.3f}s, 迭代次数: {iterations}")
    logger.info(f"最终结果: f_best={f_best:.6f}")

    return f_best, x_best, iterations, solver_time, z_vars_list, x_vars_list


if __name__ == "__main__":
    from bundle_fast.fast_g_gen import history_solution_collect
    from bundle_fast.config import get_default_config

    config = get_default_config()
    logger = get_logger("log/bundle_fast_script.log")

    mu_weights, solution_collection = history_solution_collect(config=config, logger=logger)

    f_best, x_best, iterations, solver_time, z_vars_list, x_vars_list = run(
        config=config,
        mu_weights=mu_weights,
        solution_collection=solution_collection,
        size=10,
        step=1,
        max_iterations=1000,
        tolerance=1e-5,
        realization=1,
        logger=logger,
    )

    print(f"\n最终结果:")
    print(f"f_best: {f_best}")
    print(f"x_best: {x_best}")
    print(f"iterations: {iterations}")
    print(f"solver_time: {solver_time:.3f}s")
    print(f"z_vars_list: {z_vars_list}")
    print(f"x_vars_list: {x_vars_list}")
