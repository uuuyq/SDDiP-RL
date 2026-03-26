import gurobipy as gp
from matplotlib import pyplot as plt

from bundle_RL.logger import get_logger
from sddip.sddip import ucmodelclassical, parameters


# class SubProblem:
#     """
#     求解: phi(pi) = min_x L(x, pi)
#     该类充当 Oracle，为 Master 提供函数值和子梯度。
#     """
#
#     def __init__(self, logger, problem_params, trial_point, t, n, i):
#
#     # def init_model(self


class FastModel:
    def __init__(self, logger, problem_params, trial_point, t, n, i, history_solution):
        self.logger = logger
        self.problem_params = problem_params
        self.trial_point = trial_point
        self.t = t
        self.n = n
        self.i = i
        # 子问题模型
        self.uc_bw, self.model, self.relaxed_terms, self.objective_terms = self.init_model()
        self.len = len(history_solution)
        self.history_solution = history_solution
        self.mu = []
        self.alpha_x = []
        self.alpha_z = []
        # 解析数模型需要的变量数
        sol = history_solution[0]
        self.n_vars = len(sol["z_x"]) + len(sol["z_y"]) + len(sol["z_x_bs"]) + len(sol["z_soc"]) + len(sol["x"]) + len(sol["y"]) + len(sol["x_bs"]) + len(sol["soc"])

        self.init_model()


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

        # 创建新增的变量
        for i in range(self.len):
            self.mu.append(uc_bw.model.addVar(lb=0, ub=1, vtype=gp.GRB.CONTINUOUS, name=f"mu_{i}"))
        for i in range(self.len):
            for j in range(self.n_vars):
                self.alpha_x.append(uc_bw.model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"alpha_x_{i}_{j}"))
                self.alpha_z.append(uc_bw.model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"alpha_z_{i}_{j}"))

        # 增加 mu和为1 约束
        uc_bw.model.addConstr(gp.quicksum(self.mu) == 1, "mu_sum")


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



