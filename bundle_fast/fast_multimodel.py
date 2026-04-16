import copy
from pathlib import Path

import gurobipy as gp
import numpy as np
from matplotlib import pyplot as plt

from sddip.sddip import parameters
from sddip.sddip.multimodelbuilder import MultiModelBuilder


class FastMultiModel:
    def __init__(self, logger, problem_params, trial_point, t, n, i, mu_history, alpha=100, verbose=1):
        self.logger = logger
        self.problem_params = problem_params
        self.trial_point = trial_point
        self.t = t
        self.n = n
        self.i = i
        # self.history_solution = history_solution

        self.mu_history = mu_history
        self.len = len(mu_history)

        # alpha: relaxed_sum 的权重
        self.alpha = alpha

        # Gurobi日志输出级别：0=关闭，1=开启
        self.verbose = verbose

        # 初始化变量
        self.mu = []
        self.alpha_x = []
        self.alpha_z = []
        # 创建模型
        self.model_builder, self.objective_terms = self.init_model()

    def init_model(self):
        # Get binary trial points
        x_trial_point = self.trial_point[0]
        y_trial_point = self.trial_point[1]
        x_bs_trial_point = self.trial_point[2]
        soc_trial_point = self.trial_point[3]

        # 创建MultiModelBuilder，组数等于历史解的数量
        multi_builder = MultiModelBuilder(
            self.problem_params.n_buses,
            self.problem_params.n_lines,
            self.problem_params.n_gens,
            self.problem_params.n_storages,
            self.problem_params.gens_at_bus,
            self.problem_params.storages_at_bus,
            self.problem_params.backsight_periods,
            lp_relax=False,
            n_groups=self.len
        )
        # 设置Gurobi日志输出
        multi_builder.model.setParam("OutputFlag", self.verbose)
        multi_builder = self.add_problem_constraints(
            multi_builder,
            stage=self.t,
            realization=self.n,
            iteration=0,
        )



        relaxed_sum = self.get_relaxed_sum(
            multi_builder,
            x_trial_point,
            y_trial_point,
            x_bs_trial_point,
            soc_trial_point,
            self.mu_history
        )

        # print("relaxed_sum: ", relaxed_sum)

        # 设置目标函数，使用两部分：
        # 1. relaxed_sum 的二范数（权重 alpha）
        # 2. delta 的二范数（使用权重加权）
        # 使用 r 向量各位置的绝对值约束 relaxed_sum 的每个分量
        # r = np.array([
        #     -1.27361737e-03,  1.82139936e-01,  6.96591631e-01, -1.59861276e-01,
        #     -5.99225194e-02,  3.66021115e-03,  0.00000000e+00,  0.00000000e+00,
        #      1.21268432e-01,  1.21268432e-01,  8.19133680e-01,  8.19133680e-01,
        #     -2.17355022e+00
        # ])
        # for j, rj in enumerate(r):
        #     bound = abs(float(rj))
        #     if bound > 1e-12:
        #         multi_builder.model.addConstr(relaxed_sum[j] <= bound, name=f"relaxed_upper_{j}")
        #         multi_builder.model.addConstr(relaxed_sum[j] >= -bound, name=f"relaxed_lower_{j}")


        # 范数约束
        l2_norm_relaxed = gp.quicksum(xi * xi for x in relaxed_sum for xi in x)
        multi_builder.model.addConstr(l2_norm_relaxed <= 1)

        # 计算所有组的 delta² 的加权和
        l2_norm_delta = gp.quicksum(
            self.mu_history[i] * multi_builder.variables['delta'][i] * multi_builder.variables['delta'][i]
            for i in range(self.len)
        )

        # 组合目标函数
        objective = self.alpha * l2_norm_delta
        multi_builder.model.setObjective(objective, gp.GRB.MINIMIZE)

        # 返回模型和相关组件
        return multi_builder, relaxed_sum

    def add_problem_constraints(self,
        multi_builder: MultiModelBuilder,
        stage: int,
        realization: int,
        iteration: int,
    ) -> MultiModelBuilder:
        # 为每个组添加约束
        for group_id in range(self.len):

            # 添加平衡约束
            multi_builder.add_balance_constraints(
                sum(self.problem_params.p_d[stage][realization]),
                sum(self.problem_params.re[stage][realization]),
                self.problem_params.eff_dc,
                group_id=group_id
            )

            # 添加电力流约束
            multi_builder.add_power_flow_constraints(
                self.problem_params.ptdf,
                self.problem_params.pl_max,
                self.problem_params.p_d[stage][realization],
                self.problem_params.re[stage][realization],
                self.problem_params.eff_dc,
                group_id=group_id
            )

            # 添加储能约束
            multi_builder.add_storage_constraints(
                self.problem_params.rc_max,
                self.problem_params.rdc_max,
                self.problem_params.soc_max,
                group_id=group_id
            )

            # 添加最终SOC约束（如果是最后一个阶段）
            if stage == self.problem_params.n_stages - 1:
                multi_builder.add_final_soc_constraints(
                    self.problem_params.init_soc_trial_point,
                    group_id=group_id
                )

            # 添加SOC传递约束
            multi_builder.add_soc_transfer(self.problem_params.eff_c, group_id=group_id)

            # 添加发电机约束
            multi_builder.add_generator_constraints(
                self.problem_params.pg_min, self.problem_params.pg_max,
                group_id=group_id
            )

            # 添加启停约束
            multi_builder.add_startup_shutdown_constraints(group_id=group_id)

            # 添加爬坡率约束
            multi_builder.add_ramp_rate_constraints(
                self.problem_params.r_up,
                self.problem_params.r_down,
                self.problem_params.r_su,
                self.problem_params.r_sd,
                group_id=group_id
            )

            # 添加最小启停时间约束
            multi_builder.add_up_down_time_constraints(
                self.problem_params.min_up_time, self.problem_params.min_down_time,
                group_id=group_id
            )

            # 添加切割下界
            multi_builder.add_cut_lower_bound(self.problem_params.cut_lb[stage])

        return multi_builder

    def get_relaxed_sum(
        self,
        multi_builder,
        x_trial_point,
        y_trial_point,
        x_bs_trial_point,
        soc_trial_point,
        mu,
    ):

        # # 创建新增的变量
        # for i in range(self.len):
        #     self.mu.append(multi_builder.model.addVar(lb=0, ub=1, vtype=gp.GRB.CONTINUOUS, name=f"mu_{i}"))

        all_relaxed_terms = []
        for group_id in range(self.len):
            # 添加复制约束
            relaxed_terms = multi_builder.relaxed_terms_calculate_without_binary(
                x_trial_point,
                y_trial_point,
                x_bs_trial_point,
                soc_trial_point,
                group_id=group_id
            )

            all_relaxed_terms.append(relaxed_terms)
        n_terms = len(all_relaxed_terms[0])
        relaxed_sum = [0.0] * n_terms
        for m, relaxed_terms in zip(mu, all_relaxed_terms):
            relaxed_sum = [
                s + m * term for s, term in zip(relaxed_sum, relaxed_terms)
            ]

        return all_relaxed_terms

    def get_subgradients(self):
        """获取解，z_x"""

        def flatten_to_list(nested_data):
            """递归展开任何嵌套的 list/tuple/ndarray"""
            flat = []
            for item in nested_data:
                if isinstance(item, (list, tuple, np.ndarray)):
                    flat.extend(flatten_to_list(item))
                else:
                    flat.append(item)
            return flat

        self.model_builder.model.update()
        self.model_builder.model.optimize()


        self.logger.info(f"obj: {self.model_builder.model.getObjective().getValue()}")

        self.logger.info(f"model opt status: {self.model_builder.model.status}")

        trial_point_flat = np.array(flatten_to_list(self.trial_point))

        if self.model_builder.model.status == gp.GRB.OPTIMAL:
            subgradients = []

            for group_id in range(self.len):
                group_vars = self.model_builder._get_group_variables(group_id)
                solution = []
                solution += [var.X for var in group_vars['z_x']]
                solution += [var.X for var in group_vars['z_y']]
                solution += [var.X for bs_vars in group_vars['z_x_bs'] for var in bs_vars]
                solution += [var.X for var in group_vars['z_soc']]
                solution_array = np.array(solution)
                # 计算梯度 g_t = x_t-1 - z_x
                subgradients.append(trial_point_flat - solution_array)
            return subgradients
        else:
            raise Exception(f"model status: {self.model_builder.model.status}")



# 使用示例
# if __name__ == "__main__":
#     # 示例参数（实际使用时应该从problem_params获取）
#     example_params = parameters.ProblemParams(
#         n_buses=10,
#         n_lines=15,
#         n_gens=5,
#         n_storages=2,
#         n_stages=3,
#         gens_at_bus=[...],  # 实际使用时需要正确设置
#         storages_at_bus=[...],  # 实际使用时需要正确设置
#         backsight_periods=[...],  # 实际使用时需要正确设置
#         cost_coeffs=[...],  # 实际使用时需要正确设置
#         p_d=[...],  # 实际使用时需要正确设置
#         re=[...],  # 实际使用时需要正确设置
#         eff_dc=[...],  # 实际使用时需要正确设置
#         ptdf=[...],  # 实际使用时需要正确设置
#         pl_max=[...],  # 实际使用时需要正确设置
#         rc_max=[...],  # 实际使用时需要正确设置
#         rdc_max=[...],  # 实际使用时需要正确设置
#         soc_max=[...],  # 实际使用时需要正确设置
#         init_soc_trial_point=[...],  # 实际使用时需要正确设置
#         pg_min=[...],  # 实际使用时需要正确设置
#         pg_max=[...],  # 实际使用时需要正确设置
#         r_up=[...],  # 实际使用时需要正确设置
#         r_down=[...],  # 实际使用时需要正确设置
#         r_su=[...],  # 实际使用时需要正确设置
#         r_sd=[...],  # 实际使用时需要正确设置
#         min_up_time=[...],  # 实际使用时需要正确设置
#         min_down_time=[...],  # 实际使用时需要正确设置
#         cut_lb=[...],  # 实际使用时需要正确设置
#         eff_c=[...],  # 实际使用时需要正确设置
#     )
#
#     # 示例试验点（实际使用时应该从trial_point获取）
#     example_trial_point = [
#         [0.5, 0.3, 0.7, 0.2, 0.6],  # x_trial_point
#         [100, 150, 120, 80, 200],    # y_trial_point
#         [[0.8, 0.6], [0.9, 0.7], [0.5, 0.4], [0.3, 0.2], [0.7, 0.5]],  # x_bs_trial_point
#         [0.8, 0.6]                  # soc_trial_point
#     ]
#
#     # 历史解（示例）
#     history_solution = [
#         {"z_x": [0.5, 0.3, 0.7, 0.2, 0.6], "z_y": [100, 150, 120, 80, 200],
#          "z_x_bs": [[0.8, 0.6], [0.9, 0.7], [0.5, 0.4], [0.3, 0.2], [0.7, 0.5]],
#          "z_soc": [0.8, 0.6], "x": [0.5, 0.3, 0.7, 0.2, 0.6],
#          "y": [100, 150, 120, 80, 200], "x_bs": [[0.8, 0.6], [0.9, 0.7], [0.5, 0.4], [0.3, 0.2], [0.7, 0.5]],
#          "soc": [0.8, 0.6]}
#     ]
#
#     # 创建并初始化模型
#     fast_model = FastMultiModel(
#         problem_params=example_params,
#         trial_point=example_trial_point,
#         t=0,
#         n=0,
#         i=0,
#         history_solution=history_solution
#     )
#
#     # 求解模型
#     fast_model.solve()
#
#     # 获取解
#     solution = fast_model.get_solution()
#     group_solutions = fast_model.get_group_solutions()
#
#     print("Solution:", solution)
#     print("Group solutions:", group_solutions)