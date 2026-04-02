from pathlib import Path

import numpy as np

from bundle_RL.logger import get_logger
from bundle_fast.fast_multimodel import FastMultiModel
from bundle_fast.lag_problem import LagrangianMaster
from sddip.sddip import parameters

from bundle_RL.lag_problem import SubProblem, MasterProblem
from pathlib import Path

import numpy as np

from bundle_RL.lag_problem import SubProblem, MasterProblem
from bundle_RL.logger import get_logger
from bundle_fast.fast_multimodel import FastMultiModel
from bundle_fast.lag_problem import LagrangianMaster
from sddip.sddip import parameters


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



# t = 1
# n_vars = 13
# x_trial = [1.0, 1.0, 1.0]
# y_trial = [71.52627531002818, 59.02627531002818, 66.52627531002818]
# x_bs_trial = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
# soc_trial = [5.0]
t = 5
n_vars = 13
x_trial = [-0.0, 1.0, 1.0]
y_trial = [0.0, 131.60809087723158, 45.0]
x_bs_trial = [[-0.0, 0.0], [1.0, 1.0], [1.0, 1.0]]
soc_trial = [0.0]
path = Path(r"D:\tools\workspace_pycharm\sddip-main-zou\data\01_test_cases\case6ww\t06_n06")
problem_params = parameters.Parameters(path)



def history_solution_collect():
    logger = get_logger("log/bundle_solver.log")
    realization = 0

    sub = SubProblem(logger, problem_params, trial_point=(x_trial, y_trial, x_bs_trial, soc_trial), t=t, n=realization, i=0)
    logger.info(f"场景，pd：{problem_params.p_d[t][realization]}")
    logger.info(f"场景，re：{problem_params.re[t][realization]}")

    master = MasterProblem(logger, n_vars, tolerance=1e-5)

    delta_history = []
    solution_collection = []
    # 初始化
    x_new = np.zeros(n_vars)
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

    with open(f"solutions_{t}_{realization}.json", "w", encoding="utf-8") as f:
        json.dump(solution_collection, f, ensure_ascii=False, indent=4)  # indent 使文件更易读
    with open(f"solutions_{t}_{realization}.pkl", "wb") as f:
        pickle.dump(solution_collection, f)

        # 1. 获取现有 cuts
    current_cuts = master.cuts_storage  # 或者使用 get_all_cuts(master)

    # 2. 构造对偶求解器
    lag_master = LagrangianMaster(
        logger=logger,
        n_vars=n_vars,
        cuts_storage=current_cuts,
        x_best=master.x_best,
        u=master.u
    )

    # 3. 求解得到新的 pi 和 乘子 mu
    pi_candidate, mu_weights = lag_master.solve()

    print(f"得到的对偶乘子为: {mu_weights}")
    print(f"下一个试验点 pi 为: {pi_candidate}")

    with open(f"mu_{t}_{realization}.pkl", "wb") as f:
        pickle.dump(mu_weights, f)

    return mu_weights


def bundle_fast(mu_weights, size):
    logger = get_logger("log/bundle_fast.log")
    mu_array = np.array(mu_weights)
    n_total = len(mu_array)

    # 1. 如果现有的权重数量本身就小于等于 size，直接归一化返回
    if n_total > size:

        top_indices = np.argsort(mu_array)[-size:][::-1]
        top_weights = mu_array[top_indices]
        sum_top = top_weights.sum()
        if sum_top > 1e-12:
            mu_array = top_weights / sum_top
        else:
            # 如果前 size 个权重全是 0，则平均分配（兜底逻辑）
            mu_array = np.ones(size) / size


    realization = 1
    fast = FastMultiModel(logger, problem_params, trial_point=(x_trial, y_trial, x_bs_trial, soc_trial), t=t, n=realization, i=0
                   , mu_history=mu_array)
    logger.info(f"场景，pd：{problem_params.p_d[t][realization]}")
    logger.info(f"场景，re：{problem_params.re[t][realization]}")

    subgradients = fast.get_subgradients()

    np.set_printoptions(
        precision=6,  # 保留4位小数
        suppress=True,  # 禁用科学计数法
        linewidth=100,  # 每行字符宽度，防止错行
        edgeitems=5  # 数组过长时，开头和结尾显示的元素个数
    )

    # subgradients_np = np.array(subgradients)
    # print(f"subgradients:\n{subgradients_np}")
    # print(f"Shape: {subgradients_np.shape}")

    print(f"subgradients: {subgradients}")
    print(f"len(subgradients): {len(subgradients)}")



if __name__ == "__main__":
    mu_weight = history_solution_collect()
    bundle_fast(mu_weight, 10)