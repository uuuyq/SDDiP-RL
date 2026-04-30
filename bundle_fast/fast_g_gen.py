from pathlib import Path

import numpy as np

from bundle_fast.lag_problem import SubProblem, MasterProblem
from bundle_fast.logger import get_logger
from bundle_fast.fast_multimodel import FastMultiModel
from bundle_fast.lag_problem import LagrangianMaster
from sddip.sddip import parameters
from bundle_fast import config


def get_solution_x_z(result: list, sub: SubProblem):
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




def history_solution_collect(realization = 0):
    logger = get_logger("log/bundle_solver.log")


    sub = SubProblem(logger, config.PROBLEM_PARAMS, trial_point=(config.X_TRIAL, config.Y_TRIAL, config.X_BS_TRIAL, config.SOC_TRIAL), t=config.T, n=realization, i=0)
    logger.info(f"场景，pd：{config.PROBLEM_PARAMS.p_d[config.T][realization]}")
    logger.info(f"场景，re：{config.PROBLEM_PARAMS.re[config.T][realization]}")

    master = MasterProblem(logger, config.N_VARS, tolerance=1e-5)

    delta_history = []
    solution_collection = []
    # 初始化
    x_new = np.zeros(config.N_VARS)
    g_new, f_new = sub.solve(x_new)
    # 获取解 x z_x
    get_solution_x_z(solution_collection, sub)
    master.update_strategy(x_new, f_new, g_new, ub=None)

    for i in range(200):
        master.add_cut(x_new, f_new, g_new)  # pi, sub_obj, subgradient
        ub, x_new = master.solve_master()
        g_new, f_new = sub.solve(x_new)
        # 获取解 x z_x
        get_solution_x_z(solution_collection, sub)
        serious_step, delta, stop_flag = master.update_strategy(x_new, f_new, g_new, ub)
        delta_history.append(delta)
        logger.info(f"delta: {delta}")
        if stop_flag:
            break
    import json
    import pickle

    logger.info(f"收集到 {len(solution_collection)} 个解: {solution_collection}")

    with open(f"fast_g_gen/solutions_{config.T}_{realization}.json", "w", encoding="utf-8") as f:
        json.dump(solution_collection, f, ensure_ascii=False, indent=4)  # indent 使文件更易读

        # 1. 获取现有 cuts
    current_cuts = master.cuts_storage  # 或者使用 get_all_cuts(master)

    # 2. 构造对偶求解器
    lag_master = LagrangianMaster(
        logger=logger,
        n_vars=config.N_VARS,
        cuts_storage=current_cuts,
        x_best=master.x_best,
        u=master.u
    )

    # 3. 求解得到新的 pi 和 乘子 mu
    pi_candidate, mu_weights = lag_master.solve()

    # 计算乘子加权梯度 r = mu_weights * subgradients
    gradients = np.array([cut[0] for cut in current_cuts])
    r = mu_weights @ gradients
    print(f"r (mu * subgradients):\n{r}")
    print(f"r Shape: {r.shape}")

    with open(f"fast_g_gen/mu_raw_{config.T}_{realization}.json", "w", encoding="utf-8") as f:
        json.dump(mu_weights.tolist(), f, ensure_ascii=False, indent=4)  # indent 使文件更易读


    return mu_weights, solution_collection


def bundle_fast(mu_weights, solution_collection, size, realization = 1):
    logger = get_logger("log/bundle_fast.log")
    mu_array = np.array(mu_weights)
    n_total = len(mu_array)

    selected_solution_collection = solution_collection

    # 如果权重数量大于 size，筛选 top-size 权重，并同步筛选对应历史解（保持顺序对应）
    if n_total > size:
        top_indices = np.argsort(mu_array)[-size:][::-1]
        top_weights = mu_array[top_indices]

        # 同步筛选历史解，顺序与 top_weights 完全一致
        selected_solution_collection = [solution_collection[idx] for idx in top_indices]

        sum_top = top_weights.sum()
        if sum_top > 1e-12:
            mu_array = top_weights / sum_top
        else:
            # 如果前 size 个权重全是 0，则平均分配（兜底逻辑）
            mu_array = np.ones(size) / size
    # mu_array = np.array([0.1] * 10)
    print("mu_weight: ", mu_array)
    print("solution_collection: ", selected_solution_collection)


    fast = FastMultiModel(
        logger,
        config.PROBLEM_PARAMS,
        trial_point=(config.X_TRIAL, config.Y_TRIAL, config.X_BS_TRIAL, config.SOC_TRIAL),
        t=config.T,
        n=realization,
        i=0,
        mu_history=mu_array,
        solution_collection=selected_solution_collection,
    )
    logger.info(f"场景，pd：{config.PROBLEM_PARAMS.p_d[config.T][realization]}")
    logger.info(f"场景，re：{config.PROBLEM_PARAMS.re[config.T][realization]}")

    subgradients = fast.get_subgradients()

    np.set_printoptions(
        precision=6,  # 保留4位小数
        suppress=True,  # 禁用科学计数法
        linewidth=100,  # 每行字符宽度，防止错行
        edgeitems=5  # 数组过长时，开头和结尾显示的元素个数
    )

    subgradients_np = np.array(subgradients)
    print(f"subgradients:\n{subgradients_np}")
    print(f"Shape: {subgradients_np.shape}")

    return subgradients, mu_array.tolist()


if __name__ == "__main__":
    import json

    mu_weight, solution_collection = history_solution_collect()

    subgradients, mu_list = bundle_fast(mu_weight, solution_collection, 10)

    # 保存 subgradients 和 mu 到文件
    save_data = {
        "subgradients": np.array(subgradients).tolist(),
        "mu_weights": mu_list
    }

    with open("fast_g_gen/subgradients_mu.json", "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4)
    print("已保存 subgradients 和 mu_weights 到 subgradients_mu.json")