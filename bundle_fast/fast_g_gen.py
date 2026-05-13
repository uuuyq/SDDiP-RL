"""
fast_g_gen.py

提供历史解收集和 subgradients 生成的方法
"""

import numpy as np

from bundle_fast.lag_problem import SubProblem, MasterProblem, LagrangianMaster
from bundle_fast.fast_multimodel import FastMultiModel
from bundle_fast.config import BundleConfig


def ensure_numeric(obj):
    """递归确保所有值都是数值类型"""
    if isinstance(obj, dict):
        return {k: ensure_numeric(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_numeric(item) for item in obj]
    elif isinstance(obj, (int, float)):
        return obj
    else:
        # 如果是numpy类型或其他，转换为float
        return float(obj)


def get_solution_x_z(result: list, sub: SubProblem):
    """从子问题中提取解"""
    uc = sub.uc_bw
    
    z_x = [uc.z_x[j].x for j in range(len(uc.z_x))]
    z_y = [uc.z_y[j].x for j in range(len(uc.z_y))]
    z_x_bs = [uc.z_x_bs[g][k].x for g in range(len(uc.z_x_bs)) for k in range(len(uc.z_x_bs[g]))]
    z_soc = [uc.z_soc[i].x for i in range(len(uc.z_soc))]
    x = [uc.x[j].x for j in range(len(uc.x))]
    y = [uc.y[j].x for j in range(len(uc.y))]
    x_bs = [uc.x_bs[g][k].x for g in range(len(uc.x_bs)) for k in range(len(uc.x_bs[g]))]
    soc = [uc.soc[i].x for i in range(len(uc.soc))]
    
    # 提取计算 objective_terms 所需的额外变量
    s_up = [uc.s_up[j].x for j in range(len(uc.s_up))]
    s_down = [uc.s_down[j].x for j in range(len(uc.s_down))]
    ys_p = uc.ys_p.x
    ys_n = uc.ys_n.x
    socs_p = [uc.socs_p[i].x for i in range(len(uc.socs_p))]
    socs_n = [uc.socs_n[i].x for i in range(len(uc.socs_n))]
    # x_bs_p 和 x_bs_n 是二维列表 [generator][backsight_period]
    x_bs_p = [uc.x_bs_p[g][k].x for g in range(len(uc.x_bs_p)) for k in range(len(uc.x_bs_p[g]))]
    x_bs_n = [uc.x_bs_n[g][k].x for g in range(len(uc.x_bs_n)) for k in range(len(uc.x_bs_n[g]))]
    delta = uc.delta.x
    theta = uc.theta.x
    
    result.append(
        {
            "z_x": z_x,
            "z_y": z_y,
            "z_x_bs": z_x_bs,
            "z_soc": z_soc,
            "x": x,
            "y": y,
            "x_bs": x_bs,
            "soc": soc,
            "s_up": s_up,
            "s_down": s_down,
            "ys_p": ys_p,
            "ys_n": ys_n,
            "socs_p": socs_p,
            "socs_n": socs_n,
            "x_bs_p": x_bs_p,
            "x_bs_n": x_bs_n,
            "delta": delta,
            "theta": theta,
        }
    )


def history_solution_collect(
    config: BundleConfig,
    realization: int = 0,
    max_iterations: int = 200,
    logger=None,
):
    """
    收集历史解并计算 mu 权重

    Args:
        config: BundleConfig 配置对象
        realization: 场景索引
        logger: 日志器

    Returns:
        mu_weights: mu 权重数组
        solution_collection: 历史解列表
        sg_results: SolverResults对象（包含obj_value, multipliers, n_iterations, solver_time）
    """
    from bundle_fast.lag_problem import SolverResults
    import time
    
    trial_point = config.trial_point
    start_time = time.time()

    sub = SubProblem(logger, config, realization)
    logger.info(f"历史解收集: realization={realization}, pd={config.PROBLEM_PARAMS.p_d[config.T][realization]}, re={config.PROBLEM_PARAMS.re[config.T][realization]}")

    master = MasterProblem(logger, config.N_VARS, tolerance=1e-5)

    solution_collection = []
    x_new = np.zeros(config.N_VARS)
    g_new, f_new = sub.solve(x_new)
    get_solution_x_z(solution_collection, sub)
    master.update_strategy(x_new, f_new, g_new, ub=None)

    n_iterations = 0
    for i in range(max_iterations):
        master.add_cut(x_new, f_new, g_new)
        ub, x_new = master.solve_master()
        g_new, f_new = sub.solve(x_new)
        get_solution_x_z(solution_collection, sub)
        serious_step, delta, stop_flag = master.update_strategy(x_new, f_new, g_new, ub)
        n_iterations = i + 1
        logger.info(f"迭代 {i+1}: delta={delta:.6f}")
        if stop_flag:
            break

    logger.info(f"收集到 {len(solution_collection)} 个历史解")

    # 获取现有 cuts
    current_cuts = master.cuts_storage

    # 构造对偶求解器
    lag_master = LagrangianMaster(
        logger=logger,
        n_vars=config.N_VARS,
        cuts_storage=current_cuts,
        x_best=master.x_best,
        u=master.u
    )

    # 求解得到新的 pi 和 乘子 mu
    pi_candidate, mu_weights = lag_master.solve()

    # 计算乘子加权梯度
    gradients = np.array([cut[0] for cut in current_cuts])
    r = mu_weights @ gradients
    logger.info(f"mu 加权梯度 r: {r}")

    # 使用 x_best 再求解一次子问题，获取最终的 subgradient 和 obj_value
    final_subgradient, final_obj_value = sub.solve(master.x_best)
    
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
        n_iterations=n_iterations,
        solver_time=solver_time
    )

    return mu_weights, solution_collection, sg_results


def gen_subgradient(
    config: BundleConfig,
    mu_weights,
    solution_collection,
    size,
    realization: int = 1,
    logger=None,
):
    """
    生成 subgradients

    Args:
        config: BundleConfig 配置对象
        mu_weights: mu 权重数组
        solution_collection: 历史解列表
        size: 筛选的 top solution 数量
        realization: 场景索引
        logger: 日志器

    Returns:
        subgradients: 梯度列表
        mu_list: 权重列表
        z_vars_list: z 变量列表
        x_vars_list: x 变量列表
    """
    mu_array = np.array(mu_weights)
    n_total = len(mu_array)

    selected_solution_collection = solution_collection

    # 如果权重数量大于 size，筛选 top-size 权重，并同步筛选对应历史解
    if n_total > size:
        top_indices = np.argsort(mu_array)[-size:][::-1]
        top_weights = mu_array[top_indices]
        selected_solution_collection = [solution_collection[idx] for idx in top_indices]
        sum_top = top_weights.sum()
        if sum_top > 1e-12:
            mu_array = top_weights / sum_top
        else:
            mu_array = np.ones(size) / size

    logger.info(f"bundle_fast: realization={realization}, pd={config.PROBLEM_PARAMS.p_d[config.T][realization]}, re={config.PROBLEM_PARAMS.re[config.T][realization]}")
    logger.info(f"mu_array: {mu_array}")

    trial_point = config.trial_point

    fast = FastMultiModel(
        logger,
        config,
        n=realization,
        mu_history=mu_array,
        solution_collection=selected_solution_collection,
    )

    subgradients, z_vars_list, x_vars_list = fast.get_subgradients()

    logger.info(f"subgradients shape: {np.array(subgradients).shape}")
    logger.info(f"z_vars_list: {z_vars_list}")
    logger.info(f"x_vars_list: {x_vars_list}")

    return subgradients, mu_array.tolist(), z_vars_list, x_vars_list


if __name__ == "__main__":
    import json

    from bundle_fast.logger import get_logger
    from bundle_fast.config import get_default_config

    logger = get_logger("log/fast_g_gen.log")
    config = get_default_config()

    # 步骤1: 历史解收集
    mu_weights, solution_collection, _ = history_solution_collect(config=config, realization=0, logger=logger)

    # 保存历史解
    with open(f"fast_g_gen/solutions_{config.T}_0.json", "w", encoding="utf-8") as f:
        json.dump(solution_collection, f, ensure_ascii=False, indent=4)

    # 保存 mu
    with open(f"fast_g_gen/mu_raw_{config.T}_0.json", "w", encoding="utf-8") as f:
        json.dump(mu_weights.tolist(), f, ensure_ascii=False, indent=4)

    logger.info("已保存历史解和 mu")

    # 步骤2: 生成 subgradients
    subgradients, mu_list, z_vars_list, x_vars_list = gen_subgradient(
        config=config,
        mu_weights=mu_weights,
        solution_collection=solution_collection,
        size=10,
        realization=1,
        logger=logger,
    )

    # 保存 subgradients 和 mu
    save_data = {
        "subgradients": np.array(subgradients).tolist(),
        "mu_weights": mu_list,
        "z_vars_list": z_vars_list,
        "x_vars_list": x_vars_list,
    }
    # print(save_data)  # 注释掉，避免输出大量数据

    with open("fast_g_gen/subgradients_mu.json", "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4)

    logger.info("已保存 subgradients, mu, z_vars_list, x_vars_list")
