import logging
from pathlib import Path

import logging
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from bundle_RL.logger import get_logger
from sddip.sddip import parameters

from bundle_RL.lag_problem import SubProblem, MasterProblem

logger = logging.getLogger(__name__)


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




def bundle_test():
    logger = get_logger("bundle_solver.log")

    t = 0
    n = 0
    n_vars = 13
    x_trial = [1.0, 1.0, 1.0]
    y_trial = [71.52627531002818, 59.02627531002818, 66.52627531002818]
    x_bs_trial = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    soc_trial = [5.0]
    path = Path(r"D:\tools\workspace_pycharm\sddip-main-zou\data\01_test_cases\case6ww\t06_n06")
    problem_params = parameters.Parameters(path)

    sub = SubProblem(logger, problem_params, trial_point=(x_trial, y_trial, x_bs_trial, soc_trial), t=t, n=n, i=0)
    logger.info(f"场景，pd：{problem_params.p_d[t][n]}")
    logger.info(f"场景，re：{problem_params.re[t][n]}")

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

    with open(f"solutions_{t}_{n}.json", "w", encoding="utf-8") as f:
        json.dump(solution_collection, f, ensure_ascii=False, indent=4)  # indent 使文件更易读
    with open(f"solutions_{t}_{n}.pkl", "wb") as f:
        pickle.dump(solution_collection, f)


if __name__ == "__main__":
    bundle_test()