import copy
import logging
from pathlib import Path

import numpy as np
import gurobipy as gp
from matplotlib import pyplot as plt

from bundle_RL.logger import get_logger
from sddip.sddip import ucmodelclassical, parameters

logger = logging.getLogger(__name__)


class SubProblem:
    """
    求解: phi(pi) = min_x L(x, pi)
    该类充当 Oracle，为 Master 提供函数值和子梯度。
    """

    def __init__(self, logger, problem_params, trial_point, t, n, i):
        self.logger = logger
        self.problem_params = problem_params
        self.trial_point = trial_point
        self.t = t
        self.n = n
        self.i = i
        # 子问题模型
        self.uc_bw, self.model, self.relaxed_terms, self.objective_terms = self.init_model()
    def init_model(self):
        # Get binary trial points
        x_trial_point = self.trial_point[0]
        y_trial_point = self.trial_point[1]
        x_bs_trial_point = self.trial_point[2]
        soc_trial_point = self.trial_point[3]

        # Build backward model
        uc_bw = ucmodelclassical.ClassicalModel(
            self.problem_params.n_buses,
            self.problem_params.n_lines,
            self.problem_params.n_gens,
            self.problem_params.n_storages,
            self.problem_params.gens_at_bus,
            self.problem_params.storages_at_bus,
            self.problem_params.backsight_periods,
        )

        uc_bw: ucmodelclassical.ClassicalModel = (
            self.add_problem_constraints(uc_bw, self.t, self.n, self.i)
        )

        uc_bw.relaxed_terms_calculate_without_binary(
            x_trial_point,
            y_trial_point,
            x_bs_trial_point,
            soc_trial_point,
        )

        objective_terms = uc_bw.objective_terms
        relaxed_terms = uc_bw.relaxed_terms
        uc_bw.disable_output()

        return uc_bw, uc_bw.model, relaxed_terms, objective_terms


    def add_problem_constraints(
        self,
        model_builder: ucmodelclassical.ClassicalModel,
        stage: int,
        realization: int,
        iteration: int,
    ) -> ucmodelclassical.ClassicalModel:
        model_builder.add_objective(self.problem_params.cost_coeffs)

        model_builder.add_balance_constraints(
            sum(self.problem_params.p_d[stage][realization]),
            sum(self.problem_params.re[stage][realization]),
            self.problem_params.eff_dc,
        )

        model_builder.add_power_flow_constraints(
            self.problem_params.ptdf,
            self.problem_params.pl_max,
            self.problem_params.p_d[stage][realization],
            self.problem_params.re[stage][realization],
            self.problem_params.eff_dc,
        )

        model_builder.add_storage_constraints(
            self.problem_params.rc_max,
            self.problem_params.rdc_max,
            self.problem_params.soc_max,
        )

        if stage == self.problem_params.n_stages - 1:
            model_builder.add_final_soc_constraints(
                self.problem_params.init_soc_trial_point
            )
        model_builder.add_soc_transfer(self.problem_params.eff_c)

        model_builder.add_generator_constraints(
            self.problem_params.pg_min, self.problem_params.pg_max
        )

        model_builder.add_startup_shutdown_constraints()

        model_builder.add_ramp_rate_constraints(
            self.problem_params.r_up,
            self.problem_params.r_down,
            self.problem_params.r_su,
            self.problem_params.r_sd,
        )

        model_builder.add_up_down_time_constraints(
            self.problem_params.min_up_time, self.problem_params.min_down_time
        )

        model_builder.add_cut_lower_bound(self.problem_params.cut_lb[stage])

        # TODO: cuts constrains
        # if stage < self.problem_params.n_stages - 1 and iteration > 0:
        #     if common.CutType.LAGRANGIAN in self.cut_types_added:
        #         lagrangian_coefficients = self.cc_storage.get_stage_result(
        #             stage
        #         )
        #         model_builder.add_cut_constraints_without_binary(
        #             lagrangian_coefficients[ResultKeys.ci_key],
        #             lagrangian_coefficients[ResultKeys.cg_key],
        #         )
        #     if bool(
        #         self.cut_types_added
        #         & {common.CutType.BENDERS, common.CutType.STRENGTHENED_BENDERS}
        #     ):
        #         benders_coefficients = self.bc_storage.get_stage_result(stage)
        #         model_builder.add_benders_cuts_without_binary(
        #             benders_coefficients[ResultKeys.bc_intercept_key],
        #             benders_coefficients[ResultKeys.bc_gradient_key],
        #             benders_coefficients[ResultKeys.bc_trial_point_key],
        #         )

        return model_builder

    def solve(self, pi: np.ndarray, time_limit: float | None = None)-> tuple[np.ndarray, float]:
        """
        输入: pi (当前对偶变量/乘子)
        返回: phi (函数值), g (子梯度)
        """

        gradient_len = len(self.relaxed_terms)

        total_objective = self.objective_terms + gp.quicksum(
            self.relaxed_terms[i] * pi[i] for i in range(gradient_len)
        )

        self.model.setObjective(total_objective)

        self.model.update()

        if time_limit is not None:
            self.model.setParam(
                "TimeLimit",
                max(time_limit * 60, 10),
                # Ensure that Gurobi has enough time to find at least a
                # feasible point. Otherwise, retrieving the variable
                # values would fail.
            )
        else:
            self.model.setParam("TimeLimit", gp.GRB.INFINITY)

        self.model.optimize()

        # self.solver_time += self.model.Runtime

        subgradient = np.array([t.getValue() for t in self.relaxed_terms])
        opt_value = self.model.getObjective().getValue()

        return (subgradient, opt_value)

"""
这里说明符号的含义：
ub : master求解得到的上界
f_best 下界（稳定中心对应的子问题最优目标函数值）
f_new 当前sub求解得到的目标函数值
x_new 求解master得到的新的pi值
x_best f_best对应的pi值  （稳定中心）
serious step 说明此次pi值的更新带来了足够好的f_new，可以更新f_best和x_best

算法的逻辑： master -> sub -> add_cut -> weight_update
第一次迭代中：使用 x = 0, 求解sub, 得到f_new, subgradient, add_cut, 初始化f_best=f_new
后续迭代中：
    - master求解得到f_hat (问题的上界，预测值), x_new (新的试验点)
    - sub 求解得到 f_new, g_new, 注意需要记录对应的x_new 
    - add_cut 使用f_new, g_new，以及对应的x_new
    - weight_update：
        - delta = f_hat - self.f_best (f_hat代表上界预测值，f_best是目前求解的最好的真实值，delta代表后续可以改进的大小)
        - serious_step = (f_new - self.f_best) 
            (f_new代表在x_new上面计算的真实值，f_best代表之前最好的真实值，用来判断此次移动x所带来的好处)
            -true 则可以更新f_best和x_best

"""
class MasterProblem:
    def __init__(self, logger, n_vars, tolerance):
        self.logger = logger
        self.n_vars = n_vars
        self.u = 1
        self.u_min = 0.1  # > 0
        self.m_l = 0.2  # (0, 0.5)
        self.m_r = 0.5  # (m_l, 1)
        self.i_u = 0
        self.var_est = 10 ** 9
        self.cuts_storage = []
        self.tolerance = tolerance

        self.x_best = np.zeros(n_vars)
        self.f_best = -1e9
        # self.f_hat = 0.0  # 上一次 Master 求解的预测目标值
        self.iter_idx = 0

        # --- 初始化 Gurobi 模型 ---
        self.model = gp.Model("Master_Bundle")
        self.model.setParam("OutputFlag", 0)
        self.v = self.model.addVar(lb=-gp.GRB.INFINITY, name="v")
        self.x_vars = self.model.addVars(n_vars, lb=-gp.GRB.INFINITY, name="x")  # pi

    def solve_master(self):
        """
        求解主问题 (Master)
        根据当前的 u 和 x_best，通过现有的切平面寻找下一个候选点 pi
        :return 返回新的pi值
        """
        # 设置目标函数: obj = v - u/2 * ||x - x_best||^2
        obj = self.v - self.u / 2 * gp.quicksum(
            (self.x_vars[j] - self.x_best[j]) ** 2 for j in range(self.n_vars)
        )
        self.model.setObjective(obj, gp.GRB.MAXIMIZE)
        self.model.optimize()
        self.logger.info(self.model.status)


        x_candidate = np.array([self.x_vars[j].x for j in range(self.n_vars)])
        # 返回预测值（问题的上界）和候选点
        return self.v.x, x_candidate

    def add_cut(self, x_new, f_new, g_new):
        self.iter_idx += 1
        cut_expr = f_new + gp.quicksum(
            g_new[j] * (self.x_vars[j] - x_new[j]) for j in range(self.n_vars)
        )
        self.cuts_storage.append((g_new, x_new, f_new))
        self.model.addConstr(self.v <= cut_expr, name=f"cut_{self.iter_idx}")

    def update_strategy(self, x_new, f_new, g_new, ub=None):
        """
        更新权重和状态 (Weight Update)
        根据子问题的真实反馈 f_new 和 Master 的预测值 f_hat 进行判定。
        """
        stop_flag = False
        if self.iter_idx == 0:
            self.f_best = f_new
            self.x_best = copy.copy(x_new)
            # 此时初始化，还没有进行master求解，没有ub
            return None, None, None

        # 计算预测增益 delta  上界-best下界
        delta = max(ub - self.f_best, 0)

        # Check stopping criterion
        # self.logger.info(f"delta: {delta} tolerance: {self.tolerance}")
        if delta <= self.tolerance:
            stop_flag = True
            self.logger.info(f"算法已满足终止条件, delta: {delta} tolerance: {self.tolerance}")
            # TODO 需要停止？

        # 判定 Serious Step
        serious_step = (f_new - self.f_best) >= self.m_l * delta

        # 执行原算法的 weight_update 逻辑
        self.u, self.i_u, self.var_est = self._weight_update_logic(
            self.u, self.i_u, self.var_est,
            x_new, f_new, self.x_best, self.f_best,
            ub, g_new, serious_step
        )

        # 如果是 Serious Step，更新中心点
        if serious_step:
            self.x_best = copy.copy(x_new)
            self.f_best = f_new

        return serious_step, delta, stop_flag

    def _weight_update_logic(self, u_current, i_u, var_est, x_new, f_new, x_best, f_best, f_hat, subgradient,
                             serious_step):
        """这里完整保留你提供的原 weight_update 逻辑代码"""
        variation_estimate = var_est
        delta = f_hat - f_best  # f_hat就是ub
        u_int = 2 * u_current * (1 - (f_new - f_best) / delta) if abs(delta) > 1e-12 else u_current
        u = u_current

        if serious_step:
            weight_too_large = (f_new - f_best) >= (self.m_r * delta)
            if weight_too_large and i_u > 0:
                u = u_int
            elif i_u > 3:
                u = u_current / 2
            u_new = max(u, u_current / 10, self.u_min)
            variation_estimate = max(variation_estimate, 2 * delta)
            i_u = max(i_u + 1, 1) if u_new == u_current else 1
        else:
            p = -u_current * (np.array(x_new) - np.array(x_best))
            alpha = delta - np.linalg.norm(p, ord=2) ** 2 / u_current
            variation_estimate = min(variation_estimate, np.linalg.norm(p, ord=1) + alpha)
            linearization_error = (f_new + np.array(subgradient).dot(np.array(x_best) - np.array(x_new)) - f_best)
            if (linearization_error > max(variation_estimate, 10 * delta) and i_u < -3):
                u = u_int
            u_new = min(u, 10 * u_current)
            i_u = min(i_u - 1, -1) if u_new == u_current else -1

        return u_new, i_u, variation_estimate




def bundle_test():
    logger = get_logger("log/bundle_solver.log")

    t = 0
    n_vars = 13
    x_trial = [1.0, 1.0, 1.0]
    y_trial = [71.52627531002818, 59.02627531002818, 66.52627531002818]
    x_bs_trial = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    soc_trial = [5.0]
    path = Path(r"D:\tools\workspace_pycharm\sddip-main-zou\data\01_test_cases\case6ww\t06_n06")
    problem_params = parameters.Parameters(path)

    sub = SubProblem(logger, problem_params, trial_point=(x_trial, y_trial, x_bs_trial, soc_trial), t=t, n=0, i=0)
    master = MasterProblem(logger, n_vars, tolerance=1e-5)

    delta_history = []

    x_new = np.zeros(n_vars)

    g_new, f_new = sub.solve(x_new)
    master.update_strategy(x_new, f_new, g_new, ub=None)

    for i in range(200):
        master.add_cut(x_new, f_new, g_new)  # pi, sub_obj, subgradient
        ub, x_new = master.solve_master()
        g_new, f_new = sub.solve(x_new)
        serious_step, delta, stop_flag = master.update_strategy(x_new, f_new, g_new, ub)
        delta_history.append(delta)
        logger.info(f"delta: {delta}")
        if stop_flag:
            break

    plt.figure(figsize=(8, 5))
    # 绘制折线
    plt.plot(range(len(delta_history)), delta_history,
             marker='o', linestyle='-', color='#1f77b4',
             linewidth=1.5, markersize=4, label='$\Delta$ (Convergence Gap)')
    plt.ylabel('Delta Value')
    plt.xlabel('Iteration Step')
    plt.title('Bundle Method Convergence (Delta)')
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()

    # 自动调整布局并显示
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    bundle_test()
