import logging
from pathlib import Path
from time import time

import gurobipy as gp
import numpy as np
from scipy import linalg, stats

from sddip.sddip import common

from . import (
    dualsolver,
    parameters,
    scenarios,
    sddip_logging,
    storage,
    ucmodelclassical,
    utils,
)
from .constants import ResultKeys



class Algorithm:
    def __init__(
        self,
        path: Path,
        log_dir: str,
        dual_solver: dualsolver.BundleMethod,
        mylog_dir: str
    ) -> None:
        # Logger
        self.runtime_logger = sddip_logging.RuntimeLogger(log_dir)

        # Problem specific parameters
        self.problem_params = parameters.Parameters(path)

        # Algorithm paramters
        self.n_binaries = 10
        self.error_threshold = 10 ** (-1)
        self.max_n_binaries = 10
        # Absolute change in lower bound
        self.no_improvement_tolerance = 10 ** (-8)
        self.stop_stabilization_count = 5
        self.time_limit_minutes = 5 * 60
        self.n_samples_final_ub = 150

        # Helper objects
        self.binarizer = utils.Binarizer()
        self.sc_sampler = scenarios.ScenarioSampler(
            self.problem_params.n_stages,
            self.problem_params.n_realizations_per_stage[1],
        )

        self.dual_solver = dual_solver

        self.primary_cut_mode = common.CutType.STRENGTHENED_BENDERS
        self.secondary_cut_mode = common.CutType.LAGRANGIAN

        self.n_samples_primary = 3
        self.n_samples_secondary = 1

        self.cut_types_added = set()

        # Result storage
        self.ps_storage = storage.ResultStorage(
            ResultKeys.primal_solution_keys, "primal_solutions"
        )
        self.ds_storage = storage.ResultStorage(
            ResultKeys.dual_solution_keys, "dual_solutions"
        )
        self.cc_storage = storage.ResultStorage(
            ResultKeys.cut_coefficient_keys, "cut_coefficients"
        )
        self.bc_storage = storage.ResultStorage(
            ResultKeys.benders_cut_keys, "benders_cuts"
        )
        self.dual_solver_storage = storage.ResultStorage(
            ResultKeys.dual_solver_keys, "dual_solver"
        )
        self.bound_storage = storage.ResultStorage(
            ResultKeys.bound_keys, "bounds"
        )

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)  # 设置日志等级（DEBUG/INFO/WARNING/ERROR/CRITICAL）
        # 创建一个输出到控制台的 handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        # 设置输出格式
        formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s - %(message)s')
        console_handler.setFormatter(formatter)
        # 把 handler 添加到 logger
        self.logger.addHandler(console_handler)
        # --- 新增：创建一个输出到文件的 handler ---
        # filename: 日志文件名, mode: 'a' 代表追加（append），'w' 代表每次运行覆盖
        file_handler = logging.FileHandler(mylog_dir, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)



    def fixed_binary_approximation(self) -> None:
        self.bin_multipliers = {
            "x": [1] * self.problem_params.n_gens,
            "x_bs": [
                [1] * self.problem_params.backsight_periods[g]
                for g in range(self.problem_params.n_gens)
            ],
        }

        continuous_variables_approx_error = []

        y_bin_multipliers = []
        self.y_0_bin = []
        for p_max, p_init in zip(
            self.problem_params.pg_max,
            self.problem_params.init_y_trial_point,
            strict=False,
        ):
            y_bin_multipliers.append(
                self.binarizer.calc_binary_multipliers_from_n_binaries(
                    p_max, self.n_binaries
                )
            )

            prec = self.binarizer.calc_precision_from_n_binaries(
                p_max, self.n_binaries
            )

            continuous_variables_approx_error.append(
                self.binarizer.calc_max_abs_error(prec)
            )

            self.y_0_bin += self.binarizer.get_best_binary_approximation(
                p_init, y_bin_multipliers[-1]
            )[0]

        soc_bin_multipliers = []
        self.soc_0_bin = []
        for s_max, soc_init in zip(
            self.problem_params.soc_max,
            self.problem_params.init_soc_trial_point,
            strict=False,
        ):
            soc_bin_multipliers.append(
                self.binarizer.calc_binary_multipliers_from_n_binaries(
                    s_max, self.n_binaries
                )
            )

            prec = self.binarizer.calc_precision_from_n_binaries(
                s_max, self.n_binaries
            )

            continuous_variables_approx_error.append(
                self.binarizer.calc_max_abs_error(prec)
            )

            self.soc_0_bin += self.binarizer.get_best_binary_approximation(
                soc_init, soc_bin_multipliers[-1]
            )[0]

        self.logger.info(
            "Approximation errors: %s", continuous_variables_approx_error
        )

        self.bin_multipliers["y"] = y_bin_multipliers
        self.bin_multipliers["soc"] = soc_bin_multipliers

    def run(self, n_iterations: int) -> None:
        self.logger.info("#### SDDiP-Algorithm started ####")
        self.runtime_logger.start()
        self.dual_solver.runtime_logger.start()
        # self.current_cut_mode = self.primary_cut_mode
        self.current_cut_mode = self.secondary_cut_mode  # 固定为lag
        self.n_samples = self.n_samples_primary
        lower_bounds = []
        lagrangian_cut_iterations = []

        # self.fixed_binary_approximation()

        for i in range(n_iterations):
            self.logger.info("Iteration %s", i + 1)

            ########################################
            # Cut mode selection
            ########################################
            # self.select_cut_mode(i, lower_bounds)

            ########################################
            # Sampling
            ########################################
            sampling_start_time = time()
            n_samples = self.n_samples
            samples = self.sc_sampler.generate_samples(n_samples)
            self.logger.info("Samples: %s", samples)
            self.runtime_logger.log_task_end(
                f"sampling_i{i+1}", sampling_start_time
            )

            ########################################
            # Forward pass
            ########################################
            forward_pass_start_time = time()
            v_opt_k = self.forward_pass(i, samples)
            self.runtime_logger.log_task_end(
                f"forward_pass_i{i+1}", forward_pass_start_time
            )

            ########################################
            # Statistical upper bound
            ########################################
            upper_bound_start_time = time()
            v_upper_l, v_upper_r = self.statistical_upper_bound(
                v_opt_k, n_samples
            )
            self.logger.info("Statistical upper bound: %s", v_upper_l)
            self.runtime_logger.log_task_end(
                f"upper_bound_i{i+1}", upper_bound_start_time
            )

            ########################################
            # Forward pass 足够多个路径计算上界
            ########################################
            n_samples_statistical = 300
            forward_pass_statistical_start_time = time()
            samples_statistical = self.sc_sampler.generate_samples(n_samples_statistical)
            v_opt_k = self.forward_pass_statistical(i, samples_statistical)
            self.runtime_logger.log_task_end(
                f"forward_pass_statistical_i{i + 1}", forward_pass_statistical_start_time
            )
            upper_bound_start_time = time()
            v_upper_l, v_upper_r = self.statistical_upper_bound(
                v_opt_k, n_samples_statistical
            )
            self.logger.info("paths: %s, Statistical upper bound: %s", n_samples_statistical, v_upper_l)
            self.runtime_logger.log_task_end(
                f"upper_bound_statistical_i{i + 1}", upper_bound_start_time
            )

            ########################################
            # Backward pass
            ########################################
            backward_pass_start_time = time()
            if self.current_cut_mode == common.CutType.LAGRANGIAN:
                lagrangian_cut_iterations.append(i)
                self.backward_pass(i + 1, samples)
                self.cut_types_added.update([common.CutType.LAGRANGIAN])
            elif self.current_cut_mode in [
                common.CutType.BENDERS,
                common.CutType.STRENGTHENED_BENDERS,
            ]:
                self.backward_benders(i + 1, samples)
                self.cut_types_added.update(
                    [
                        common.CutType.BENDERS,
                        common.CutType.STRENGTHENED_BENDERS,
                    ]
                )
            self.runtime_logger.log_task_end(
                f"backward_pass_i{i+1}", backward_pass_start_time
            )

            ########################################
            # Lower bound
            ########################################
            lower_bound_start_time = time()
            v_lower = self.lower_bound_without_binary(i + 1)
            lower_bounds.append(v_lower)
            self.logger.info(f"Lower bound: {v_lower} ")
            self.runtime_logger.log_task_end(
                f"lower_bound_i{i+1}", lower_bound_start_time
            )

            bound_dict = self.bound_storage.create_empty_result_dict()
            bound_dict[ResultKeys.lb_key] = v_lower
            bound_dict[ResultKeys.ub_l_key] = v_upper_l
            bound_dict[ResultKeys.ub_r_key] = v_upper_r
            self.bound_storage.add_result(i, 0, 0, bound_dict)

            ########################################
            # Stopping criteria
            ########################################
            # Stop if time limit reached
            if (
                time() - self.runtime_logger.global_start_time
                >= self.time_limit_minutes * 60
            ):
                break
            # Stop if lower bound stagnates
            stagnation = False
            if (
                len(lagrangian_cut_iterations)
                >= (self.stop_stabilization_count + 1)
                and i > 0
            ):
                stagnation = (
                    lower_bounds[-1]
                    - lower_bounds[-(self.stop_stabilization_count + 1)]
                    < self.no_improvement_tolerance
                )
            if stagnation:
                self.logger.info("Lower bound stabilized.")
                break

        self.runtime_logger.log_experiment_end()
        self.dual_solver.runtime_logger.log_experiment_end()

        ########################################
        # Final upper bound
        ########################################
        n_samples = self.n_samples_final_ub
        samples = self.sc_sampler.generate_samples(n_samples)
        v_opt_k = self.forward_pass(n_iterations, samples)
        v_upper_l, v_upper_r = self.statistical_upper_bound(v_opt_k, n_samples)

        bound_dict = self.bound_storage.create_empty_result_dict()
        bound_dict[ResultKeys.lb_key] = 0
        bound_dict[ResultKeys.ub_l_key] = v_upper_l
        bound_dict[ResultKeys.ub_r_key] = v_upper_r
        self.bound_storage.add_result(n_iterations, 0, 0, bound_dict)

        self.logger.info("#### SDDiP-Algorithm finished ####")

    def forward_pass(self, iteration: int, samples: list) -> list:
        i = iteration
        n_samples = len(samples)
        v_opt_k = []

        for k in range(n_samples):
            x_trial_point = self.problem_params.init_x_trial_point
            y_trial_point = self.problem_params.init_y_trial_point
            x_bs_trial_point = self.problem_params.init_x_bs_trial_point
            soc_trial_point = self.problem_params.init_soc_trial_point

            v_opt_k.append(0)
            for t, n in zip(
                range(self.problem_params.n_stages), samples[k], strict=False
            ):
                # y_binary_trial_multipliers = linalg.block_diag(
                #     *self.bin_multipliers["y"]
                # )
                # soc_binary_trial_multipliers = linalg.block_diag(
                #     *self.bin_multipliers["soc"]
                # )

                # Create forward model
                uc_fw = ucmodelclassical.ClassicalModel(
                    self.problem_params.n_buses,
                    self.problem_params.n_lines,
                    self.problem_params.n_gens,
                    self.problem_params.n_storages,
                    self.problem_params.gens_at_bus,
                    self.problem_params.storages_at_bus,
                    self.problem_params.backsight_periods,
                )

                # uc_fw.binary_approximation(
                #     self.bin_multipliers["y"], self.bin_multipliers["soc"]
                # )

                uc_fw: ucmodelclassical.ClassicalModel = (
                    self.add_problem_constraints(uc_fw, t, n, i)
                )

                # 新增变量x_bin_copy_vars = x_binary_trial_point
                # uc_fw.add_sddip_copy_constraints(
                #     x_trial_point,
                #     y_trial_point,
                #     x_bs_trial_point,
                #     soc_trial_point,
                # )
                # 约束 x_bin_copy_vars = z_x
                # uc_fw.add_copy_constraints(
                #     y_binary_trial_multipliers, soc_binary_trial_multipliers
                # )

                # 增加约束 relaxed_terms = x_trial_point - z_x
                uc_fw.relaxed_terms_calculate_without_binary(
                    x_trial_point,
                    y_trial_point,
                    x_bs_trial_point,
                    soc_trial_point,
                )

                # relaxed_terms = 0
                uc_fw.zero_relaxed_terms()

                # Solve problem
                uc_fw.disable_output()
                uc_fw.model.optimize()

                try:
                    x_kt = [x_g.x for x_g in uc_fw.x]
                    y_kt = [y_g.x for y_g in uc_fw.y]
                    x_bs_kt = [
                        [x_bs.x for x_bs in x_bs_g] for x_bs_g in uc_fw.x_bs
                    ]
                    soc_kt = [soc_s.x for soc_s in uc_fw.soc]
                except AttributeError:
                    uc_fw.model.write("model.lp")
                    uc_fw.model.computeIIS()
                    uc_fw.model.write("model.ilp")
                    raise

                # Value of stage t objective function
                v_value_function = uc_fw.model.getObjective().getValue()
                v_opt_kt = v_value_function - uc_fw.theta.x
                v_opt_k[-1] += v_opt_kt

                x_trial_point = x_kt
                y_trial_point = y_kt
                if any(x_bs_kt):
                    x_bs_trial_point = [
                        [x_trial_point[g]] + x_bs_kt[g][:-1]
                        for g in range(self.problem_params.n_gens)
                    ]
                soc_trial_point = soc_kt

                ps_dict = self.ps_storage.create_empty_result_dict()
                ps_dict[ResultKeys.x_key] = x_kt
                ps_dict[ResultKeys.y_key] = y_kt
                ps_dict[ResultKeys.x_bs_key] = x_bs_trial_point
                ps_dict[ResultKeys.soc_key] = soc_kt
                ps_dict[ResultKeys.v_key] = v_value_function
                ps_dict[ResultKeys.theta_key] = uc_fw.theta.x

                self.ps_storage.add_result(i, k, t, ps_dict)

        return v_opt_k
    def forward_pass_statistical(self, iteration: int, samples: list) -> list:
        """
        obj 统计
        """
        i = iteration
        n_samples = len(samples)
        v_opt_k = []

        for k in range(n_samples):
            x_trial_point = self.problem_params.init_x_trial_point
            y_trial_point = self.problem_params.init_y_trial_point
            x_bs_trial_point = self.problem_params.init_x_bs_trial_point
            soc_trial_point = self.problem_params.init_soc_trial_point

            v_opt_k.append(0)
            for t, n in zip(
                range(self.problem_params.n_stages), samples[k], strict=False
            ):
                # y_binary_trial_multipliers = linalg.block_diag(
                #     *self.bin_multipliers["y"]
                # )
                # soc_binary_trial_multipliers = linalg.block_diag(
                #     *self.bin_multipliers["soc"]
                # )

                # Create forward model
                uc_fw = ucmodelclassical.ClassicalModel(
                    self.problem_params.n_buses,
                    self.problem_params.n_lines,
                    self.problem_params.n_gens,
                    self.problem_params.n_storages,
                    self.problem_params.gens_at_bus,
                    self.problem_params.storages_at_bus,
                    self.problem_params.backsight_periods,
                )

                # uc_fw.binary_approximation(
                #     self.bin_multipliers["y"], self.bin_multipliers["soc"]
                # )

                uc_fw: ucmodelclassical.ClassicalModel = (
                    self.add_problem_constraints(uc_fw, t, n, i)
                )

                # 新增变量x_bin_copy_vars = x_binary_trial_point
                # uc_fw.add_sddip_copy_constraints(
                #     x_trial_point,
                #     y_trial_point,
                #     x_bs_trial_point,
                #     soc_trial_point,
                # )
                # 约束 x_bin_copy_vars = z_x
                # uc_fw.add_copy_constraints(
                #     y_binary_trial_multipliers, soc_binary_trial_multipliers
                # )

                # 增加约束 relaxed_terms = x_trial_point - z_x
                uc_fw.relaxed_terms_calculate_without_binary(
                    x_trial_point,
                    y_trial_point,
                    x_bs_trial_point,
                    soc_trial_point,
                )

                # relaxed_terms = 0
                uc_fw.zero_relaxed_terms()

                # Solve problem
                uc_fw.disable_output()
                uc_fw.model.optimize()

                try:
                    x_kt = [x_g.x for x_g in uc_fw.x]
                    y_kt = [y_g.x for y_g in uc_fw.y]
                    x_bs_kt = [
                        [x_bs.x for x_bs in x_bs_g] for x_bs_g in uc_fw.x_bs
                    ]
                    soc_kt = [soc_s.x for soc_s in uc_fw.soc]
                except AttributeError:
                    uc_fw.model.write("model.lp")
                    uc_fw.model.computeIIS()
                    uc_fw.model.write("model.ilp")
                    raise

                # Value of stage t objective function
                v_value_function = uc_fw.model.getObjective().getValue()


                v_opt_kt = v_value_function - uc_fw.theta.x
                v_opt_k[-1] += v_opt_kt

                x_trial_point = x_kt
                y_trial_point = y_kt
                if any(x_bs_kt):
                    x_bs_trial_point = [
                        [x_trial_point[g]] + x_bs_kt[g][:-1]
                        for g in range(self.problem_params.n_gens)
                    ]
                soc_trial_point = soc_kt

                # ps_dict = self.ps_storage.create_empty_result_dict()
                # ps_dict[ResultKeys.x_key] = x_kt
                # ps_dict[ResultKeys.y_key] = y_kt
                # ps_dict[ResultKeys.x_bs_key] = x_bs_trial_point
                # ps_dict[ResultKeys.soc_key] = soc_kt
                # ps_dict[ResultKeys.v_key] = v_value_function
                #
                # self.ps_storage.add_result(i, k, t, ps_dict)

        return v_opt_k

    def statistical_upper_bound(self, v_opt_k: list, n_samples: int) -> float:
        v_mean = np.mean(v_opt_k)
        v_std = np.std(v_opt_k)
        alpha = 0.05

        v_upper_l = v_mean + stats.norm.ppf(alpha / 2) * v_std / np.sqrt(
            n_samples
        )
        v_upper_r = v_mean - stats.norm.ppf(alpha / 2) * v_std / np.sqrt(
            n_samples
        )

        return v_upper_l, v_upper_r

    def select_cut_mode(self, iteration: int, lower_bounds: list) -> None:
        # no_improvement_condition = False
        #
        # if iteration > 1:
        #     delta = max((lower_bounds[-1] - lower_bounds[-2]), 0)
        #     no_improvement_condition = delta <= self.no_improvement_tolerance
        #
        # if self.current_cut_mode == self.secondary_cut_mode:
        #     self.current_cut_mode = self.primary_cut_mode
        #     self.n_samples = self.n_samples_primary
        # elif no_improvement_condition:
        #     self.current_cut_mode = self.secondary_cut_mode
        #     self.n_samples = self.n_samples_secondary
        self.current_cut_mode = self.secondary_cut_mode
        self.n_samples = self.n_samples_secondary

    # def backward_pass(self, iteration: int, samples: list) -> None:
    #     i = iteration
    #     n_samples = len(samples)
    #
    #     for t in reversed(range(1, self.problem_params.n_stages)):
    #         for k in range(n_samples):
    #             n_realizations = self.problem_params.n_realizations_per_stage[
    #                 t
    #             ]
    #             ds_dict = self.ds_storage.create_empty_result_dict()
    #             cc_dict = self.cc_storage.create_empty_result_dict()
    #             dual_solver_dict = (
    #                 self.dual_solver_storage.create_empty_result_dict()
    #             )
    #
    #             for n in range(n_realizations):
    #                 # Get binary trial points
    #                 print(i - 1, k, t - 1)
    #                 y_trial_point = self.ps_storage.get_result(
    #                     i - 1, k, t - 1
    #                 )[ResultKeys.y_key]
    #                 x_trial_point = self.ps_storage.get_result(
    #                     i - 1, k, t - 1
    #                 )[ResultKeys.x_key]
    #                 x_bs_trial_point = self.ps_storage.get_result(
    #                     i - 1, k, t - 1
    #                 )[ResultKeys.x_bs_key]
    #                 soc_trial_point = self.ps_storage.get_result(
    #                     i - 1, k, t - 1
    #                 )[ResultKeys.soc_key]
    #                 theta_trial_point = self.ps_storage.get_result(
    #                     i - 1, k, t - 1
    #                 )[ResultKeys.theta_key]
    #
    #                 # Build backward model
    #                 uc_bw = ucmodelclassical.ClassicalModel(
    #                     self.problem_params.n_buses,
    #                     self.problem_params.n_lines,
    #                     self.problem_params.n_gens,
    #                     self.problem_params.n_storages,
    #                     self.problem_params.gens_at_bus,
    #                     self.problem_params.storages_at_bus,
    #                     self.problem_params.backsight_periods,
    #                 )
    #
    #                 # uc_bw.binary_approximation(
    #                 #     self.bin_multipliers["y"], self.bin_multipliers["soc"]
    #                 # )
    #
    #                 uc_bw: ucmodelclassical.ClassicalModel = (
    #                     self.add_problem_constraints(uc_bw, t, n, i)
    #                 )
    #
    #                 uc_bw.relaxed_terms_calculate_without_binary_normalized(
    #                     x_trial_point,
    #                     y_trial_point,
    #                     x_bs_trial_point,
    #                     soc_trial_point,
    #                     theta_trial_point,
    #                     self.problem_params.cost_coeffs,
    #                 )
    #
    #                 objective_terms = 0
    #                 relaxed_terms = uc_bw.relaxed_terms
    #
    #                 # 注意需要将relaxed_terms中的x_trial_point - z_x转化成z_x - x_trial_point
    #                 for j in range(len(relaxed_terms) - 1):
    #                     relaxed_terms[j] = -relaxed_terms[j]
    #
    #
    #                 # if t == 1 and n == 1:
    #                 #     self.logger.info(relaxed_terms)
    #                 #     uc_bw.model.write("model.lp")
    #
    #                 uc_bw.disable_output()
    #
    #                 trial_point = (
    #                     x_trial_point
    #                     + y_trial_point
    #                     + [
    #                         x_bs_g
    #                         for x_bs in x_bs_trial_point
    #                         for x_bs_g in x_bs
    #                     ]
    #                     + soc_trial_point
    #                 )
    #
    #                 self.len_trial_points = len(trial_point)
    #
    #                 _, sg_results = self.dual_solver.solve(
    #                     uc_bw.model,
    #                     objective_terms,
    #                     relaxed_terms,
    #                     normalization = True
    #                 )
    #                 dual_multipliers = sg_results.multipliers.tolist()
    #                 pi0 = dual_multipliers[-1]
    #                 pi = dual_multipliers[:-1]
    #
    #                 if pi0 < 1e-6:
    #                     continue
    #                 print(f"pi0: {pi0} pi: {pi}")
    #                 # dual_value = sg_results.obj_value - np.array(
    #                 #     dual_multipliers
    #                 # ).dot(trial_point)
    #                 dual_value = sg_results.obj_value + pi0 * theta_trial_point + sum(
    #                     [
    #                         pi[j] * x_trial_point[j]
    #                         for j in range(len(x_trial_point))
    #                     ]
    #                 )
    #
    #                 pi = [-pi[j] / pi0 for j in range(len(pi))]
    #                 intercept = dual_value / pi0
    #
    #
    #
    #                 # print(f"pi: {pi} intercept: {intercept}")
    #
    #                 # Dual value and multiplier for each realization
    #                 ds_dict[ResultKeys.dv_key].append(intercept)
    #                 ds_dict[ResultKeys.dm_key].append(pi)
    #
    #                 dual_solver_dict[ResultKeys.ds_iterations].append(
    #                     sg_results.n_iterations
    #                 )
    #                 dual_solver_dict[ResultKeys.ds_solver_time].append(
    #                     sg_results.solver_time
    #                 )
    #
    #
    #             self.ds_storage.add_result(i, k, t, ds_dict)
    #             self.dual_solver_storage.add_result(i, k, t, dual_solver_dict)
    #
    #             # Calculate and store cut coefficients
    #             if len(ds_dict[ResultKeys.dv_key]) == 0:
    #                 intercept = 0
    #                 gradient = [0] * self.len_trial_points
    #             else:
    #                 intercept = np.mean(np.array(ds_dict[ResultKeys.dv_key]), axis=0).tolist()
    #                 gradient = np.mean(np.array(ds_dict[ResultKeys.dm_key]), axis=0).tolist()
    #
    #
    #             # probabilities = self.problem_params.prob[t]
    #             # intercept = np.array(probabilities).dot(
    #             #     np.array(ds_dict[ResultKeys.dv_key])
    #             # )
    #             # gradient = np.array(probabilities).dot(
    #             #     np.array(ds_dict[ResultKeys.dm_key])
    #             # )
    #
    #             cc_dict[ResultKeys.ci_key] = intercept
    #             cc_dict[ResultKeys.cg_key] = gradient
    #
    #             self.cc_storage.add_result(i, k, t - 1, cc_dict)

    def backward_pass(self, iteration: int, samples: list):
        i = iteration  # 从1开始
        n_samples = len(samples)

        for t in reversed(range(1, self.problem_params.n_stages)):

            for k in range(n_samples):

                cc_dict = self.cc_storage.create_empty_result_dict()

                n_realizations = self.problem_params.n_realizations_per_stage[t]

                # Get trial points
                y_trial_point = self.ps_storage.get_result(
                    i - 1, k, t - 1
                )[ResultKeys.y_key]
                x_trial_point = self.ps_storage.get_result(
                    i - 1, k, t - 1
                )[ResultKeys.x_key]
                x_bs_trial_point = self.ps_storage.get_result(
                    i - 1, k, t - 1
                )[ResultKeys.x_bs_key]
                soc_trial_point = self.ps_storage.get_result(
                    i - 1, k, t - 1
                )[ResultKeys.soc_key]
                theta_trial = self.ps_storage.get_result(
                    i - 1, k, t - 1
                )[ResultKeys.theta_key]

                X_trial = (
                        x_trial_point
                        + y_trial_point
                        + [val for bs in x_bs_trial_point for val in bs]
                        + soc_trial_point
                )

                lag_cuts_list = []
                for n in range(n_realizations):
                    inner_model = self.create_inner_model(t, n, i, False)
                    outer_model = self.create_outer_model(X_trial, theta_trial)
                    # level bundle methods
                    pi_star, pi0_star, flag = self.level_bundle_methods(inner_model, outer_model, X_trial, theta_trial, 200)
                    if pi0_star < 1e-6 or not flag :
                        continue
                    # self.logger.info(f"t: {t} i : {i} pi_star: {pi_star}, pi0_star: {pi0_star}")
                    # cut: pi * x + pi0 * theta >= inner_model_obj + pi * x_hat + pi0 * theta_hat
                    inner_model.add_inner_objective(self.problem_params.cost_coeffs, pi_star, pi0_star)
                    inner_model.model.optimize()
                    intercept = inner_model.model.getObjective().getValue()
                    pi = -pi_star / pi0_star
                    intercept = intercept / pi0_star
                    lag_cuts_list.append(list(pi) + [intercept])
                # Calculate and store cut coefficients
                if len(lag_cuts_list) > 1:
                    lag_cuts_list = np.mean(np.array(lag_cuts_list), axis=0).tolist()
                    # print("lag_average", lag_cuts_list)
                    cc_dict[ResultKeys.ci_key] = lag_cuts_list[-1]
                    cc_dict[ResultKeys.cg_key] = lag_cuts_list[:-1]

                    self.cc_storage.add_result(i, k, t - 1, cc_dict)
                else:
                    cc_dict[ResultKeys.ci_key] = 0
                    cc_dict[ResultKeys.cg_key] = [0] * len(X_trial)

                    self.cc_storage.add_result(i, k, t - 1, cc_dict)

        return

    def create_inner_model(self, t, n, i, relax=False):
        inner_model = ucmodelclassical.ClassicalModel(
            self.problem_params.n_buses,
            self.problem_params.n_lines,
            self.problem_params.n_gens,
            self.problem_params.n_storages,
            self.problem_params.gens_at_bus,
            self.problem_params.storages_at_bus,
            self.problem_params.backsight_periods,
            lp_relax=relax  # lp_relax=True时level bundle只有一个解，False则有多解，但每个解都相同
        )

        # inner_model 添加问题约束: z_X与X的约束
        inner_model: ucmodelclassical.ClassicalModel = (
            self.add_problem_constraints(inner_model, t, n, i)
        )
        inner_model: ucmodelclassical.ClassicalModel = self.add_z_Var_constrains(inner_model)
        return inner_model
    def add_z_Var_constrains(self, inner_model: ucmodelclassical.ClassicalModel):
        inner_model.add_z_var_constrains(self.problem_params.soc_max, self.problem_params.pg_min, self.problem_params.pg_max)
        return inner_model

    def create_outer_model(self, X_trial, theta_trial, benders_pi_list=None):
        from .outermodel import OuterModel
        outer_model = OuterModel(
            len(X_trial), X_trial, theta_trial, benders_pi_list=benders_pi_list
        )
        return outer_model

    def level_bundle_methods(
            self,
            inner_model,
            outer_model,
            X_trial,
            theta_trial,
            iteration_limit,
            level_factor=0.3,
            atol=1e-2,
            rtol=1e-2,
            pi0Coef=1e-2,
            timeLimit=60 * 60
    ):

        subgradient_list = []
        pi_hat = np.ones(len(X_trial)) * 0
        pi0_hat = 1
        pi_star = None
        pi0_star = None

        LB = float('-inf')  # LB
        UB = float('inf')  # UB
        lpiold = float("inf")
        iter = 0
        while iter < iteration_limit:
            # inner_model.model.params.PoolSearchMode = 2  # 启用多解搜索（灵活模式）
            # inner_model.model.params.PoolSolutions = 10  # 返回最多 10 个解
            # inner_model.model.params.PoolGap = 0.1  # 每个解的目标值误差不超过 5%
            # inner_model.model.setParam('MIPStart', True)  # 启用初始解

            # 更新pi_hat, pi0_hat
            z_X, obj_term = inner_model.add_inner_objective(self.problem_params.cost_coeffs, pi_hat, pi0_hat)
            inner_model.model.optimize()
            if inner_model.model.status == 3:  # 如果模型不可行
                inner_model.model.computeIIS()  # 计算不可行约束集
                inner_model.model.write("infeasible_model.ilp")  # 将不可行的约束写入文件
            if inner_model.model.status != 2:
                self.logger.info(
                    f"inner_model optimize---bundle interation:{iter}---inner_model.status:{inner_model.model.status}")
                break

            # 最优解，添加到cut_history
            # y_kt = [y_g.x for y_g in inner_model.y]
            # s_up_kt = [s_up_g.x for s_up_g in inner_model.s_up]
            # s_down_kt = [s_down_g.x for s_down_g in inner_model.s_down]
            # ys_p_kt = inner_model.ys_p.x
            # ys_n_kt = inner_model.ys_n.x
            # socs_p_kt = [socs_p_g.x for socs_p_g in inner_model.socs_p]
            # socs_n_kt = [socs_n_g.x for socs_n_g in inner_model.socs_n]
            # x_bs_p_kt = [x_bs_p_g.x for g in range(inner_model.n_generators) for x_bs_p_g in inner_model.x_bs_p[g]]
            # x_bs_n_kt = [x_bs_n_g.x for g in range(inner_model.n_generators) for x_bs_n_g in inner_model.x_bs_n[g]]
            # delta_kt = inner_model.delta.x
            # theta_kt = inner_model.theta.x
            # coefficients = self.problem_params.cost_coeffs
            # penalty = coefficients[-1]
            # coefficients = (
            #         coefficients + [penalty] * (2 * inner_model.n_storages + 2 * len(x_bs_p_kt) + 1) + [1]
            # )
            # variables = (
            #         y_kt
            #         + s_up_kt
            #         + s_down_kt
            #         + [ys_p_kt, ys_n_kt]
            #         + socs_p_kt
            #         + socs_n_kt
            #         + x_bs_p_kt
            #         + x_bs_n_kt
            #         + [delta_kt]
            #         + [theta_kt]
            # )
            # # 计算目标函数的形式计算theta
            # theta_value = sum(variables[i] * coefficients[i] for i in range(len(coefficients)))
            # # 获取z值
            # z_x_kt = [x_g.x for x_g in inner_model.z_x]
            # z_y_kt = [y_g.x for y_g in inner_model.z_y]
            # z_x_bs_kt = [
            #     x_bs.x for x_bs_g in inner_model.z_x_bs for x_bs in x_bs_g
            # ]
            # z_soc_kt = [soc_s.x for soc_s in inner_model.z_soc]
            # pi_subgradient = z_x_kt + z_y_kt + z_x_bs_kt + z_soc_kt
            # pi0_subgradient = theta_value

            pi_subgradient = [z_X[i].x for i in range(len(z_X))]
            pi0_subgradient = obj_term.getValue()
            subgradient_list.append(pi_subgradient + [pi0_subgradient])
            # 次梯度构造cut加入到outer_model中
            # print("outer_model add cut:", subgradient_list[-1])
            outer_model.add_constrains(subgradient_list[-1])
            outer_model.model.optimize()
            pi_dummy = [outer_model.pi[i].x for i in range(len(outer_model.pi))]
            pi0_dummy = outer_model.pi0.x

            if outer_model.model.status != 2:
                outer_model.model.write("outer_model.lp")
                self.logger.info(
                    f"outer_model optimize---interation:{iter}---outer_model.status:{outer_model.model.status}")

            inner_obj = inner_model.model.getObjective().getValue()
            gap = inner_obj - sum(pi_hat[i] * X_trial[i] for i in range(len(X_trial))) - pi0_hat * theta_trial
            if gap > LB:
                LB = gap
                pi_star = pi_hat.copy()
                pi0_star = pi0_hat
            outer_obj = outer_model.model.getObjective().getValue()
            UB = outer_obj
            # if iter % 50 == 0:
            # print(f"LB:{LB} UB:{UB} iter:{iter} pi_hat{pi_hat} pi0_hat{pi0_hat}")
            gapTol = 5e-3
            tol = 1e-4
            if UB - LB < gapTol * UB or UB - LB < 1e-6:
                # print(f"LB:{LB} UB:{UB} ************bundle收敛***********")
                # if pi0_star > 1e-6:
                #     print(f"pi_star+pi0_star: {pi_star} {pi0_star}")
                #     print(f"pi0_best > 1e-6")
                # else:
                #     print(f"pi_star+pi0_star: {pi_star} {pi0_star}")
                #     print(f"pi0Hat <= 1e-6")
                if pi0_star > 1e-6 and LB / pi0_star >= tol * (abs(theta_trial) + 1):
                    return pi_star, pi0_star, True

            QPsolved = True
            # level
            level = UB - level_factor * (UB - LB)
            outer_model.set_lower_bound(level)
            outer_model.set_level_obj(pi_hat, pi0_hat)

            outer_model.model.params.Method = 2
            outer_model.model.update()
            outer_model.model.optimize()
            if outer_model.model.status != 2:
                # print('QP status: ' + str(outer_model.model.status) + ' with Method=' + str(
                #     outer_model.model.params.Method) + '... Switching to 1')
                outer_model.model.params.Method = 1
                outer_model.model.update()
                outer_model.model.optimize()
                if outer_model.model.status != 2:
                    # print('QP status: ' + str(outer_model.model.status) + ' with Method=' + str(
                    #     outer_model.model.params.Method) + '... Switching to 0')
                    outer_model.model.params.Method = 0
                    outer_model.model.update()
                    outer_model.model.optimize()
                    if outer_model.model.status != 2:
                        # print('QP status: ' + str(outer_model.model.status) + ' with Method=' + str(
                        #     outer_model.model.params.Method) + '... Stop!')
                        QPsolved = False

            pi_hat_old = pi_hat.copy()
            pi0_hat_old = pi0_hat
            if QPsolved:
                for i in range(len(outer_model.pi)):
                    pi_hat[i] = outer_model.pi[i].x
                pi0_hat = outer_model.pi0.x
                lpiold = outer_model.L.x
            else:
                pi_hat = pi_dummy
                pi0_hat = pi0_dummy

            # 若未找到最优解，或当前解与上次迭代的解非常接近（小于 1e-10），则认为解已收敛
            if QPsolved == False or (max(abs(pi_hat[i] - pi_hat_old[i]) for i in range(len(pi_hat))) < 1e-10 and abs(
                    pi0_hat - pi0_hat_old) < 1e-10 and abs(outer_model.L.x - lpiold) < 1e-10):
                # print('Same Solution/QP not solved! QPsolved:', QPsolved)
                # if pi0_star > 1e-6:
                #     print('pi0_best > 1e-6')
                # 若 pi0Best 足够大并且满足界限条件，则将该情景的割平面约束添加到主问题模型中
                if pi0_star > 1e-6 and LB >= tol * (abs(theta_trial) + 1):
                    return pi_star, pi0_star, True

            # 恢复模型的目标函数、删去level约束
            outer_model.recover()
            outer_model.model.params.Method = -1  # -1是什么？？
            outer_model.model.update()

            iter = iter + 1

        return pi_star, pi0_star, False



    def backward_benders(self, iteration: int, samples: list) -> None:
        i = iteration
        n_samples = len(samples)
        for t in reversed(range(1, self.problem_params.n_stages)):
            for k in range(n_samples):
                n_realizations = self.problem_params.n_realizations_per_stage[
                    t
                ]
                bc_dict = self.bc_storage.create_empty_result_dict()
                y_binary_trial_point = self.ps_storage.get_result(
                    i - 1, k, t - 1
                )[ResultKeys.y_key]
                x_trial_point = self.ps_storage.get_result(i - 1, k, t - 1)[
                    ResultKeys.x_key
                ]
                x_bs_trial_point = self.ps_storage.get_result(i - 1, k, t - 1)[
                    ResultKeys.x_bs_key
                ]
                soc_binary_trial_point = self.ps_storage.get_result(
                    i - 1, k, t - 1
                )[ResultKeys.soc_key]

                trial_point = (
                    x_trial_point
                    + y_binary_trial_point
                    + [val for bs in x_bs_trial_point for val in bs]
                    + soc_binary_trial_point
                )

                y_binary_trial_multipliers = linalg.block_diag(
                    *self.bin_multipliers["y"]
                )
                soc_binary_trial_multipliers = linalg.block_diag(
                    *self.bin_multipliers["soc"]
                )

                dual_multipliers = []
                opt_values = []

                for n in range(n_realizations):
                    # Create forward model
                    uc_fw = ucmodelclassical.ClassicalModel(
                        self.problem_params.n_buses,
                        self.problem_params.n_lines,
                        self.problem_params.n_gens,
                        self.problem_params.n_storages,
                        self.problem_params.gens_at_bus,
                        self.problem_params.storages_at_bus,
                        self.problem_params.backsight_periods,
                        lp_relax=True,
                    )

                    uc_fw.binary_approximation(
                        self.bin_multipliers["y"], self.bin_multipliers["soc"]
                    )

                    uc_fw: ucmodelclassical.ClassicalModel = (
                        self.add_problem_constraints(uc_fw, t, n, i)
                    )

                    uc_fw.add_sddip_copy_constraints(
                        x_trial_point,
                        y_binary_trial_point,
                        x_bs_trial_point,
                        soc_binary_trial_point,
                    )

                    uc_fw.add_copy_constraints(
                        y_binary_trial_multipliers,
                        soc_binary_trial_multipliers,
                    )

                    uc_fw.model.optimize()

                    copy_constrs = uc_fw.sddip_copy_constrs

                    dm = []
                    try:
                        for constr in copy_constrs:
                            dm.append(constr.getAttr(gp.GRB.attr.Pi))
                    except AttributeError:
                        uc_fw.model.write("model.lp")
                        uc_fw.model.computeIIS()
                        uc_fw.model.write("model.ilp")
                        raise

                    dual_multipliers.append(dm)

                    if self.current_cut_mode == common.CutType.BENDERS:
                        opt_values.append(
                            uc_fw.model.getObjective().getValue()
                        )
                    elif (
                        self.current_cut_mode
                        == common.CutType.STRENGTHENED_BENDERS
                    ):
                        dual_model = ucmodelclassical.ClassicalModel(
                            self.problem_params.n_buses,
                            self.problem_params.n_lines,
                            self.problem_params.n_gens,
                            self.problem_params.n_storages,
                            self.problem_params.gens_at_bus,
                            self.problem_params.storages_at_bus,
                            self.problem_params.backsight_periods,
                        )
                        dual_model.binary_approximation(
                            self.bin_multipliers["y"],
                            self.bin_multipliers["soc"],
                        )
                        dual_model: ucmodelclassical.ClassicalModel = (
                            self.add_problem_constraints(dual_model, t, n, i)
                        )
                        dual_model.relax_sddip_copy_constraints(
                            x_trial_point,
                            y_binary_trial_point,
                            x_bs_trial_point,
                            soc_binary_trial_point,
                        )
                        dual_model.add_copy_constraints(
                            y_binary_trial_multipliers,
                            soc_binary_trial_multipliers,
                        )

                        copy_terms = dual_model.relaxed_terms

                        (
                            _,
                            dual_value,
                        ) = self.dual_solver.get_subgradient_and_value(
                            dual_model.model,
                            dual_model.objective_terms,
                            copy_terms,
                            dm,
                        )
                        opt_values.append(dual_value)

                opt_values = np.array(opt_values)
                dual_multipliers = np.array(dual_multipliers)
                v = np.average(opt_values)
                pi = np.average(dual_multipliers, axis=0)

                bc_dict[ResultKeys.bc_intercept_key] = v
                bc_dict[ResultKeys.bc_gradient_key] = pi.tolist()
                bc_dict[ResultKeys.bc_trial_point_key] = list(trial_point)

                self.bc_storage.add_result(i, k, t - 1, bc_dict)

    def lower_bound_without_binary(self, iteration: int) -> float:
        t = 0
        n = 0
        i = iteration

        x_trial_point = self.problem_params.init_x_trial_point
        y_trial_point = self.problem_params.init_y_trial_point
        x_bs_trial_point = self.problem_params.init_x_bs_trial_point
        soc_trial_point = self.problem_params.init_soc_trial_point

        # Create forward model
        uc_fw = ucmodelclassical.ClassicalModel(
            self.problem_params.n_buses,
            self.problem_params.n_lines,
            self.problem_params.n_gens,
            self.problem_params.n_storages,
            self.problem_params.gens_at_bus,
            self.problem_params.storages_at_bus,
            self.problem_params.backsight_periods,
        )

        # uc_fw.binary_approximation(
        #     self.bin_multipliers["y"], self.bin_multipliers["soc"]
        # )

        uc_fw: ucmodelclassical.ClassicalModel = self.add_problem_constraints(
            uc_fw, t, n, i
        )

        # uc_fw.add_sddip_copy_constraints(
        #     x_trial_point, y_trial_point, x_bs_trial_point, soc_trial_point
        # )
        #
        # uc_fw.add_copy_constraints(
        #     y_binary_trial_multipliers, soc_binary_trial_multipliers
        # )

        uc_fw.relaxed_terms_calculate_without_binary(
            x_trial_point,
            y_trial_point,
            x_bs_trial_point,
            soc_trial_point,
        )

        uc_fw.zero_relaxed_terms()

        # Solve problem
        uc_fw.disable_output()
        uc_fw.model.optimize()

        # Value of stage t objective function
        return uc_fw.model.getObjective().getValue()

    def lower_bound(self, iteration: int) -> float:
        t = 0
        n = 0
        i = iteration

        x_trial_point = self.problem_params.init_x_trial_point
        y_trial_point = self.y_0_bin
        x_bs_trial_point = self.problem_params.init_x_bs_trial_point
        soc_trial_point = self.soc_0_bin

        y_binary_trial_multipliers = linalg.block_diag(
            *self.bin_multipliers["y"]
        )
        soc_binary_trial_multipliers = linalg.block_diag(
            *self.bin_multipliers["soc"]
        )

        # Create forward model
        uc_fw = ucmodelclassical.ClassicalModel(
            self.problem_params.n_buses,
            self.problem_params.n_lines,
            self.problem_params.n_gens,
            self.problem_params.n_storages,
            self.problem_params.gens_at_bus,
            self.problem_params.storages_at_bus,
            self.problem_params.backsight_periods,
        )

        uc_fw.binary_approximation(
            self.bin_multipliers["y"], self.bin_multipliers["soc"]
        )

        uc_fw: ucmodelclassical.ClassicalModel = self.add_problem_constraints(
            uc_fw, t, n, i
        )

        uc_fw.add_sddip_copy_constraints(
            x_trial_point, y_trial_point, x_bs_trial_point, soc_trial_point
        )

        uc_fw.add_copy_constraints(
            y_binary_trial_multipliers, soc_binary_trial_multipliers
        )

        # Solve problem
        uc_fw.disable_output()
        uc_fw.model.optimize()

        # Value of stage t objective function
        return uc_fw.model.getObjective().getValue()

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

        if stage < self.problem_params.n_stages - 1 and iteration > 0:
            if common.CutType.LAGRANGIAN in self.cut_types_added:
                lagrangian_coefficients = self.cc_storage.get_stage_result(
                    stage
                )
                model_builder.add_cut_constraints_without_binary(
                    lagrangian_coefficients[ResultKeys.ci_key],
                    lagrangian_coefficients[ResultKeys.cg_key],
                )
            if bool(
                self.cut_types_added
                & {common.CutType.BENDERS, common.CutType.STRENGTHENED_BENDERS}
            ):
                benders_coefficients = self.bc_storage.get_stage_result(stage)
                model_builder.add_benders_cuts(
                    benders_coefficients[ResultKeys.bc_intercept_key],
                    benders_coefficients[ResultKeys.bc_gradient_key],
                    benders_coefficients[ResultKeys.bc_trial_point_key],
                )

        return model_builder
