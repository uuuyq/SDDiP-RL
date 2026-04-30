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


def bundle_fast(
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
        sg_results: SolverResults对象（包含obj_value, multipliers, n_iterations, solver_time）
    """
    from bundle_fast.lag_problem import SolverResults
    
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

    # 使用 x_best 再求解一次子问题，获取最终的 subgradient 和 obj_value
    from bundle_fast.lag_problem import SubProblem
    
    trial_point = config.trial_point
    sub = SubProblem(logger, config.PROBLEM_PARAMS, trial_point, config.T, realization, 0)
    final_subgradient, final_obj_value = sub.solve(x_best)
    
    # 计算 dual_value（需要减去 trial_point 的影响）
    trial_point_flat = np.array(
        trial_point[0] + trial_point[1] + 
        [val for bs in trial_point[2] for val in bs] + 
        trial_point[3]
    )
    dual_value = final_obj_value - final_subgradient.dot(trial_point_flat)
    
    solver_time = time.time() - start_time
    
    # 封装成 SolverResults
    sg_results = SolverResults()
    sg_results.set_values(
        obj_value=dual_value,
        multipliers=final_subgradient,
        n_iterations=iterations,
        solver_time=solver_time
    )
    
    logger.info("=" * 40)
    logger.info(f"完成! 总耗时: {solver_time:.3f}s, 迭代次数: {iterations}")
    logger.info(f"最终结果: dual_value={dual_value:.6f}")

    return sg_results


if __name__ == "__main__":
    from bundle_fast.fast_g_gen import history_solution_collect
    from bundle_fast.config import get_default_config

    config = get_default_config()
    logger = get_logger("log/bundle_fast_script.log")

    mu_weights, solution_collection = history_solution_collect(config=config, logger=logger)

    f_best, x_best, iterations, solver_time, z_vars_list, x_vars_list = bundle_fast(
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





# ============================================================================
# Bundle算法接口说明（从sddipclassical_without_binary_with_bundle_fast.py提取）
# ============================================================================
#
# 【Bundle Method 核心接口】
# 位置: sddip.sddip.dualsolver.BundleMethod
#
# 【1. 初始化接口】
# ----------------------------------------------------------------------------
# from sddip.sddip.dualsolver import BundleMethod
#
# dual_solver = BundleMethod(
#     max_iterations=1000,      # 最大迭代次数
#     tolerance=1e-5,           # 收敛容差（预测上升量<=tolerance时停止）
#     log_dir="log/",           # 日志目录
#     predicted_ascent="abs",   # 预测上升量计算方式: "abs"(绝对) 或 "rel"(相对)
#     time_limit=3600.0         # 时间限制（秒），可选
# )
#
# 关键参数:
#   - u_init = 1: 初始权重
#   - u_min = 0.1: 最小权重
#   - m_l = 0.2: serious step下界系数 (0, 0.5)
#   - m_r = 0.5: serious step上界系数 (m_l, 1)
#
#
# 【2. 求解接口 - solve()方法】
# ----------------------------------------------------------------------------
# 调用示例（在backward_pass中）:
#
#     _, sg_results = self.dual_solver.solve(
#         uc_bw.model,              # Gurobi模型对象
#         objective_terms,          # 目标函数项（Gurobi表达式）
#         relaxed_terms,            # 松弛项列表（Gurobi表达式列表）
#         normalization=False       # 是否使用标准化（可选，默认False）
#     )
#
# 参数说明:
#   - model: Gurobi Model对象，表示待优化的问题
#   - objective_terms: 原始目标函数表达式
#   - relaxed_terms: 被松弛的约束项列表（对应要生成割平面的变量）
#   - normalization: bool，是否使用L1范数标准化（添加pi0>=0和归一化约束）
#
# 返回值:
#   - tuple[gp.Model, SolverResults]
#     - model: 更新后的Gurobi模型
#     - sg_results: SolverResults对象，包含以下属性:
#       - obj_value: 最优目标值（float）
#       - multipliers: 最优对偶乘子/次梯度（numpy.ndarray）
#       - n_iterations: 迭代次数（int）
#       - solver_time: 求解时间（秒，float）
#
#
# 【3. 后处理 - 提取割平面系数】
# ----------------------------------------------------------------------------
# 在backward_pass中的使用流程:
#
#     # 1) 调用bundle求解
#     _, sg_results = self.dual_solver.solve(
#         uc_bw.model,
#         objective_terms,
#         relaxed_terms,
#         normalization=False
#     )
#
#     # 2) 提取对偶乘子（次梯度）
#     dual_multipliers = sg_results.multipliers.tolist()
#
#     # 3) 计算对偶值（需要减去trial_point的影响）
#     dual_value = sg_results.obj_value - np.array(dual_multipliers).dot(trial_point)
#
#     # 4) 存储结果
#     ds_dict[ResultKeys.dv_key].append(dual_value)      # 对偶值
#     ds_dict[ResultKeys.dm_key].append(dual_multipliers) # 对偶乘子
#
#     # 5) 记录bundle求解器信息
#     dual_solver_dict[ResultKeys.ds_iterations].append(sg_results.n_iterations)
#     dual_solver_dict[ResultKeys.ds_solver_time].append(sg_results.solver_time)
#
#
# 【4. 割平面系数聚合】
# ----------------------------------------------------------------------------
# 对所有场景（realizations）的结果进行加权平均:
#
#     probabilities = self.problem_params.prob[t]  # 各场景概率
#     intercept = np.array(probabilities).dot(
#         np.array(ds_dict[ResultKeys.dv_key])      # 所有场景的对偶值
#     )
#     gradient = np.array(probabilities).dot(
#         np.array(ds_dict[ResultKeys.dm_key])      # 所有场景的对偶乘子
#     )
#
#     # 存储割平面系数
#     cc_dict[ResultKeys.ci_key] = intercept.tolist()  # 截距
#     cc_dict[ResultKeys.cg_key] = gradient.tolist()   # 梯度
#
#
# 【5. Bundle算法内部工作流程】
# ----------------------------------------------------------------------------
# solve()方法内部逻辑:
#   1) 初始化: 获取初始次梯度和最优下界f_best
#   2) 创建子问题: create_subproblem()或create_normalized_subproblem()
#   3) 迭代求解:
#      a) 添加新的割平面到子问题: v <= f_new + g^T(x - x_new)
#      b) 设置目标: max v - u/2 * ||x - x_best||^2
#      c) 求解子问题得到候选解x_new
#      d) 调用get_subgradient_and_value()获取新的次梯度和函数值
#      e) 计算预测上升量delta = v.x - f_best
#      f) 检查收敛: delta <= tolerance则停止
#      g) Serious step判断: f_new - f_best >= m_l * delta
#         - 如果是serious step: 更新x_best和f_best
#         - 如果否: 保持x_best不变
#      h) 权重更新: 根据serious step调整proximity weight u
#   4) 返回结果: (model, SolverResults)
#
#
# 【6. 关键辅助方法】
# ----------------------------------------------------------------------------
# get_subgradient_and_value(model, objective_terms, relaxed_terms, multipliers, time_remaining)
#   - 功能: 求解原问题，获取次梯度和目标值
#   - 参数:
#     * model: Gurobi模型
#     * objective_terms: 目标函数项
#     * relaxed_terms: 松弛项列表
#     * multipliers: 对偶乘子（用于加权relaxed_terms）
#     * time_remaining: 剩余时间
#   - 返回: (subgradient: list, obj_value: float)
#
# weight_update(u_current, i_u, var_est, x_new, f_new, x_best, f_best, f_hat, subgradient, serious_step)
#   - 功能: 更新bundle方法的权重参数u
#   - 返回: (u_new, i_u_new, var_est_new)
#
# create_subproblem(n_dual_multipliers)
#   - 功能: 创建标准bundle子问题（QP）
#   - 返回: (subproblem_model, v_var, x_vars)
#
# create_normalized_subproblem(n_dual_multipliers)
#   - 功能: 创建标准化bundle子问题（带L1范数约束）
#   - 返回: (subproblem_model, v_var, x_vars)
#
#
# 【7. 与bundle_fast_script.py的对应关系】
# ----------------------------------------------------------------------------
# bundle_fast_script.py中的实现是SDDiP中BundleMethod的独立版本:
#
#   SDDiP中的BundleMethod          bundle_fast_script.py
#   -----------------------       ----------------------
#   BundleMethod.solve()      ->  run_bundle_with_cuts()
#   create_subproblem()       ->  MasterProblem（主问题）
#   get_subgradient_and_value()->  SubProblem.solve()（子问题）
#   serious step判断          ->  MasterProblem.update_strategy()
#   weight_update()           ->  （内部实现可能不同）
#
# 主要区别:
#   - SDDiP的BundleMethod直接在solve()中管理完整的bundle迭代
#   - bundle_fast_script.py将主问题和子问题分离，支持warmstart cuts
#   - bundle_fast_script.py使用预生成的cuts进行加速
#
# ============================================================================
