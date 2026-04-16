import copy
from pathlib import Path

import gurobipy as gp
import numpy as np
from matplotlib import pyplot as plt

from sddip.sddip import parameters
from sddip.sddip.multimodelbuilder_with_offset import MultiModelBuilderWithOffset


class FastMultiModel:
    def __init__(self, logger, problem_params, trial_point, t, n, i, mu_history, solution_collection, alpha=100, verbose=0):
        self.logger = logger
        self.problem_params = problem_params
        self.trial_point = trial_point
        self.t = t
        self.n = n
        self.i = i

        self.mu_history = mu_history
        self.solution_collection = solution_collection
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

    def _extract_offsets_from_solution(self, solution_collection):
        """从 solution_collection 中提取偏移值，转换为 MultiModelBuilderWithOffset 需要的格式

        solution_collection 中每个元素的结构：
        {
            "z_x": [1.0, 1.0, 1.0],           # 扁平列表
            "z_y": [71.5, 59.0, 66.5],        # 扁平列表
            "z_x_bs": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],  # 扁平列表
            "z_soc": [5.0],
            "x": [1.0, 1.0, 1.0],
            "y": [71.5, 59.0, 66.5],
            "x_bs": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            "soc": [5.0]
        }

        需要转换为：
        x_t: [[x_g1, x_g2, x_g3], ...]  # shape: (n_groups, n_generators)
        x_bs_t: [[[x_bs_g1_k1, x_bs_g1_k2], [x_bs_g2_k1, x_bs_g2_k2], ...], ...]  # shape: (n_groups, n_generators, backsight_periods[g])
        """
        n_groups = len(solution_collection)
        n_generators = self.problem_params.n_gens
        n_storages = self.problem_params.n_storages
        backsight_periods = self.problem_params.backsight_periods

        # 初始化偏移值列表
        x_t = []
        y_t = []
        x_bs_t = []
        soc_t = []
        z_t_x = []
        z_t_y = []
        z_t_x_bs = []
        z_t_soc = []

        for sol in solution_collection:
            # x_t 和 y_t 直接使用
            x_t.append(sol["x"].copy())
            y_t.append(sol["y"].copy())
            soc_t.append(sol["soc"].copy())

            # z_t_x 和 z_t_y 直接使用
            z_t_x.append(sol["z_x"].copy())
            z_t_y.append(sol["z_y"].copy())
            z_t_soc.append(sol["z_soc"].copy())

            # x_bs_t 和 z_t_x_bs 需要从扁平列表转换为二维结构
            # 扁平列表顺序：[g0_k0, g0_k1, g1_k0, g1_k1, g2_k0, g2_k1, ...]
            x_bs_flat = sol["x_bs"]
            z_x_bs_flat = sol["z_x_bs"]

            x_bs_2d = []
            z_x_bs_2d = []
            idx = 0
            for g in range(n_generators):
                n_periods = backsight_periods[g]
                x_bs_2d.append(x_bs_flat[idx:idx + n_periods])
                z_x_bs_2d.append(z_x_bs_flat[idx:idx + n_periods])
                idx += n_periods

            x_bs_t.append(x_bs_2d)
            z_t_x_bs.append(z_x_bs_2d)

        return x_t, y_t, x_bs_t, soc_t, z_t_x, z_t_y, z_t_x_bs, z_t_soc

    def init_model(self):
        # Get binary trial points
        x_trial_point = self.trial_point[0]
        y_trial_point = self.trial_point[1]
        x_bs_trial_point = self.trial_point[2]
        soc_trial_point = self.trial_point[3]

        # 从历史解中提取偏移值
        x_t, y_t, x_bs_t, soc_t, z_t_x, z_t_y, z_t_x_bs, z_t_soc = self._extract_offsets_from_solution(
            self.solution_collection
        )

        # 创建MultiModelBuilderWithOffset，组数等于历史解的数量
        multi_builder = MultiModelBuilderWithOffset(
            self.problem_params.n_buses,
            self.problem_params.n_lines,
            self.problem_params.n_gens,
            self.problem_params.n_storages,
            self.problem_params.gens_at_bus,
            self.problem_params.storages_at_bus,
            self.problem_params.backsight_periods,
            lp_relax=False,
            n_groups=self.len,
            x_t=x_t,
            y_t=y_t,
            x_bs_t=x_bs_t,
            soc_t=soc_t,
            z_t_x=z_t_x,
            z_t_y=z_t_y,
            z_t_x_bs=z_t_x_bs,
            z_t_soc=z_t_soc,
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

        # 范数约束: sum of all terms squared
        l2_norm_relaxed = gp.QuadExpr()
        for term in relaxed_sum:
            l2_norm_relaxed.add(term * term)
        multi_builder.model.addConstr(l2_norm_relaxed <= 1, "norm_relaxed")

        # 计算所有组的 delta² 的加权和
        l2_norm_delta = gp.QuadExpr()
        for i in range(self.len):
            delta_var = multi_builder.variables['delta'][i]
            l2_norm_delta.add(self.mu_history[i] * delta_var * delta_var)

        # 组合目标函数
        objective = self.alpha * l2_norm_delta
        multi_builder.model.setObjective(objective, gp.GRB.MINIMIZE)

        # 返回模型和相关组件
        return multi_builder, relaxed_sum

    def add_problem_constraints(self,
        multi_builder: MultiModelBuilderWithOffset,
        stage: int,
        realization: int,
        iteration: int,
    ) -> MultiModelBuilderWithOffset:
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

        all_relaxed_terms = []
        for group_id in range(self.len):
            # 计算松弛项
            relaxed_terms = multi_builder.relaxed_terms_calculate_without_binary(
                x_trial_point,
                y_trial_point,
                x_bs_trial_point,
                soc_trial_point,
                group_id=group_id
            )

            all_relaxed_terms.append(relaxed_terms)

        # 计算加权求和
        n_terms = len(all_relaxed_terms[0])
        relaxed_sum = [gp.LinExpr(0) for _ in range(n_terms)]
        for m, relaxed_terms in zip(mu, all_relaxed_terms):
            relaxed_sum = [
                s + m * term for s, term in zip(relaxed_sum, relaxed_terms)
            ]

        return relaxed_sum

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

        # self.logger.info(f"model opt status: {self.model_builder.model.status}")

        trial_point_flat = np.array(flatten_to_list(self.trial_point))

        if self.model_builder.model.status == gp.GRB.OPTIMAL:
            subgradients = []

            for group_id in range(self.len):
                group_vars = self.model_builder._get_group_variables(group_id)
                solution = []
                # z_x, z_y, z_x_bs, z_soc 现在是表达式 (偏移量 + alpha变量)
                # 需要用 getValue() 获取值
                for var in group_vars['z_x']:
                    if hasattr(var, 'getValue'):
                        solution.append(var.getValue())
                    else:
                        solution.append(var)
                for var in group_vars['z_y']:
                    if hasattr(var, 'getValue'):
                        solution.append(var.getValue())
                    else:
                        solution.append(var)
                for bs_vars in group_vars['z_x_bs']:
                    for var in bs_vars:
                        if hasattr(var, 'getValue'):
                            solution.append(var.getValue())
                        else:
                            solution.append(var)
                for var in group_vars['z_soc']:
                    if hasattr(var, 'getValue'):
                        solution.append(var.getValue())
                    else:
                        solution.append(var)

                solution_array = np.array(solution)
                # 计算梯度 g_t = x_t-1 - z_x
                subgradients.append(trial_point_flat - solution_array)
            return subgradients
        else:
            raise Exception(f"model status: {self.model_builder.model.status}")
