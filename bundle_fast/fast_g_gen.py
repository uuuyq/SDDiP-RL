"""
fast_g_gen.py

提供历史解收集和 subgradients 生成的方法
"""

import numpy as np

from bundle_fast.lag_problem import SubProblem, MasterProblem, LagrangianMaster
from bundle_fast.fast_multimodel import FastMultiModel
from bundle_fast.config import BundleConfig


def get_solution_x_z(result: list, sub: SubProblem):
    """从子问题中提取解"""
    z_x = [sub.uc_bw.z_x[j].x for j in range(len(sub.uc_bw.z_x))]
    z_y = [sub.uc_bw.z_y[j].x for j in range(len(sub.uc_bw.z_y))]
    z_x_bs = [sub.uc_bw.z_x_bs[g][k].x for g in range(len(sub.uc_bw.z_x_bs)) for k in range(len(sub.uc_bw.z_x_bs[g]))]
    z_soc = [sub.uc_bw.z_soc[i].x for i in range(len(sub.uc_bw.z_soc))]
    x = [sub.uc_bw.x[j].x for j in range(len(sub.uc_bw.x))]
    y = [sub.uc_bw.y[j].x for j in range(len(sub.uc_bw.y))]
    x_bs = [sub.uc_bw.x_bs[g][k].x for g in range(len(sub.uc_bw.x_bs)) for k in range(len(sub.uc_bw.x_bs[g]))]
    soc = [sub.uc_bw.soc[i].x for i in range(len(sub.uc_bw.soc))]
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
        }
    )


def history_solution_collect(
    config: BundleConfig,
    realization: int = 0,
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
    """
    trial_point = config.trial_point

    sub = SubProblem(logger, config.PROBLEM_PARAMS, trial_point=trial_point, t=config.T, n=realization, i=0)
    logger.info(f"历史解收集: realization={realization}, pd={config.PROBLEM_PARAMS.p_d[config.T][realization]}, re={config.PROBLEM_PARAMS.re[config.T][realization]}")

    master = MasterProblem(logger, config.N_VARS, tolerance=1e-5)

    solution_collection = []
    x_new = np.zeros(config.N_VARS)
    g_new, f_new = sub.solve(x_new)
    get_solution_x_z(solution_collection, sub)
    master.update_strategy(x_new, f_new, g_new, ub=None)

    for i in range(200):
        master.add_cut(x_new, f_new, g_new)
        ub, x_new = master.solve_master()
        g_new, f_new = sub.solve(x_new)
        get_solution_x_z(solution_collection, sub)
        serious_step, delta, stop_flag = master.update_strategy(x_new, f_new, g_new, ub)
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

    return mu_weights, solution_collection


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
        config.PROBLEM_PARAMS,
        trial_point=trial_point,
        t=config.T,
        n=realization,
        i=0,
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
    mu_weights, solution_collection = history_solution_collect(config=config, realization=0, logger=logger)

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

    with open("fast_g_gen/subgradients_mu.json", "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4)

    logger.info("已保存 subgradients, mu, z_vars_list, x_vars_list")
