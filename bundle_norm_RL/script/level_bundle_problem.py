"""
Level Bundle 算法求解脚本

算法逻辑:
    Level Bundle 方法通过 inner_model (子问题) 和 outer_model (主问题) 交替迭代求解:
    1. inner_model: 给定乘子 (pi_hat, pi0_hat)，求解 Lagrangian 松弛子问题，得到次梯度
    2. outer_model: 收集次梯度构造切平面，求解得到新的乘子 (pi_dummy, pi0_dummy)
    3. Level 策略: 在 outer_model 中设定 level 下界，最小化与当前中心的距离
    4. 重复直到 LB 和 UB 的 gap 足够小

对偶边界约束扩展 (通过参数控制是否开启):
    范数边界约束: 在 outer model 中添加 |π_j| ≤ B_t · w_j · π_0 或 Σ w_j |π_j| ≤ B_t · π_0
       - L1 范数边界: 限制 π 的 L∞ 范数
       - L∞ 范数边界: 限制 π 的加权 L1 范数

符号说明:
    pi_hat: 当前乘子 (pi 部分)
    pi0_hat: 当前乘子 (pi0 部分，对应 theta)
    pi_star: 最优乘子 (pi 部分)
    pi0_star: 最优乘子 (pi0 部分)
    LB: 下界 (inner_obj - pi_hat * X_trial - pi0_hat * theta_trial 的最大值)
    UB: 上界 (outer_model 的目标函数值)
    level_factor: level 策略的参数，控制 level = UB - level_factor * (UB - LB)
    B_t: 对偶边界值 (B_t 非 None 时启用范数边界约束)
    norm_bound_type: 范数边界约束类型 ("l1" 或 "linf")
    weights: 权重系数列表 (非二元化方式下默认全为 1)
"""

import logging

import numpy as np
import gurobipy as gp

from sddip.sddip import ucmodelclassical
from sddip.sddip.outermodel import OuterModel

# logger = logging.getLogger(__name__)


class SolverResults:
    """Level Bundle 求解结果封装"""
    def __init__(self):
        self.pi_star = None
        self.pi0_star = None
        self.converged = False
        self.lb = None
        self.ub = None
        self.n_iterations = None
        self.solver_time = None
        self.lb_history = []
        self.ub_history = []
        self.time_history = []

    def set_values(self, pi_star, pi0_star, converged, lb, ub, n_iterations, solver_time,
                   lb_history=None, ub_history=None, time_history=None):
        self.pi_star = pi_star
        self.pi0_star = pi0_star
        self.converged = converged
        self.lb = lb
        self.ub = ub
        self.n_iterations = n_iterations
        self.solver_time = solver_time
        self.lb_history = lb_history or []
        self.ub_history = ub_history or []
        self.time_history = time_history or []

    def toString(self):
        pi0_str = f"{self.pi0_star:.6f}" if self.pi0_star is not None else "None"
        return (
            f"pi_star: {self.pi_star}, pi0_star: {pi0_str}, "
            f"converged: {self.converged}, LB: {self.lb:.4f}, UB: {self.ub:.4f}, "
            f"iterations: {self.n_iterations}, time: {self.solver_time:.4f}s"
        )


class InnerProblem:
    """
    Inner Problem: 给定乘子 (pi_hat, pi0_hat)，求解 Lagrangian 松弛子问题

        min_{x,z} pi_hat^T * z_X + pi0_hat * obj_term

    其中 z_X = (z_x, z_y, z_x_bs, z_soc) 为 copy 变量
    """

    def __init__(self, logger, config, n, i_override=None):
        """
        Args:
            logger: 日志器
            config: LevelBundleConfig 对象
            n: realization 索引
            i_override: 可选，覆盖 config 中的 iteration
        """
        self.logger = logger
        self.config = config
        self.problem_params = config.PROBLEM_PARAMS
        self.trial_point = config.trial_point
        self.t = config.T
        self.n = n
        self.i = i_override if i_override is not None else config.iteration
        self.X_trial = config.X_trial
        # 构建 inner model
        self.uc_bw, self.model = self.init_model()

    def init_model(self):
        """初始化 inner model，包含问题约束和 z 变量约束"""
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

        # 添加 z 变量约束
        uc_bw.add_z_var_constrains(
            self.problem_params.soc_max,
            self.problem_params.pg_min,
            self.problem_params.pg_max,
        )

        uc_bw.disable_output()

        return uc_bw, uc_bw.model

    def add_problem_constraints(
        self,
        model_builder: ucmodelclassical.ClassicalModel,
        stage: int,
        realization: int,
        iteration: int,
    ) -> ucmodelclassical.ClassicalModel:
        """添加问题约束（与 lag_problem.py 中 SubProblem 相同）"""
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

        # 添加 cuts 约束
        if stage < self.problem_params.n_stages - 1 and iteration > 0:
            # 添加 Lagrangian cuts
            if self.config.dual_solver_storage is not None:
                try:
                    for s in range(self.problem_params.n_stages):
                        lagrangian_result = self.config.dual_solver_storage.get_stage_result(s)
                        if lagrangian_result and 'dm' in lagrangian_result and 'dv' in lagrangian_result:
                            cut_gradients = lagrangian_result['dm']
                            cut_intercepts = lagrangian_result['dv']
                            if cut_gradients and cut_intercepts:
                                model_builder.add_cut_constraints_without_binary(
                                    cut_intercepts,
                                    cut_gradients,
                                )
                except Exception as e:
                    self.logger.warning(f"Failed to add Lagrangian cuts: {e}")

            # 添加 Benders cuts
            if self.config.bc_storage is not None:
                try:
                    for s in range(self.problem_params.n_stages):
                        benders_result = self.config.bc_storage.get_stage_result(s)
                        if benders_result and 'bc_gradient' in benders_result and 'bc_intercept' in benders_result:
                            cut_gradients = benders_result['bc_gradient']
                            cut_intercepts = benders_result['bc_intercept']
                            trial_points = [self.trial_point[0] + self.trial_point[1] +
                                           [val for bs in self.trial_point[2] for val in bs] +
                                           self.trial_point[3]] * len(cut_gradients)
                            if cut_gradients and cut_intercepts:
                                model_builder.add_benders_cuts_without_binary(
                                    cut_intercepts,
                                    cut_gradients,
                                    trial_points,
                                )
                except Exception as e:
                    self.logger.warning(f"Failed to add Benders cuts: {e}")

        return model_builder

    def solve(self, pi_hat, pi0_hat):
        """
        给定乘子 (pi_hat, pi0_hat)，设置 inner objective 并求解

        返回:
            z_X_values, obj_term_value, inner_obj
        """
        z_X, obj_term = self.uc_bw.add_inner_objective(
            self.problem_params.cost_coeffs, pi_hat, pi0_hat
        )

        self.model.optimize()

        if self.model.status == 3:
            self.model.computeIIS()
            self.model.write("inner_infeasible_model.ilp")
            raise RuntimeError("Inner model is infeasible")

        if self.model.status != 2:
            self.logger.warning(
                f"Inner model status: {self.model.status}, not optimal"
            )
            return None, None, None

        z_X_values = [z_X[i].x for i in range(len(z_X))]
        obj_term_value = obj_term.getValue()
        inner_obj = self.model.getObjective().getValue()

        return z_X_values, obj_term_value, inner_obj


class OuterProblem:
    """
    Outer Problem: 收集次梯度，构造切平面，求解得到新的乘子

    最大化: L - pi^T * X_trial - pi0 * theta_trial
    约束:
        L <= pi^T * subgradient_pi + pi0 * subgradient_pi0  (切平面)
        ||pi||_1 + pi0 <= 1  (归一化约束)
        |π_j| ≤ B_t · w_j · π_0  (L1 范数边界约束, 限制 L∞, B_t 非 None 时启用)
        或 Σ w_j |π_j| ≤ B_t · π_0  (L∞ 范数边界约束, 限制加权 L1, B_t 非 None 时启用)
    """

    def __init__(self, logger, dim_pi, X_trial, theta_trial,
                 B_t=None, norm_bound_type="l1", weights=None):
        """
        Args:
            logger: 日志器
            dim_pi: pi 的维度
            X_trial: trial point (拼接后)
            theta_trial: theta 的 trial point
            B_t: 对偶边界值 (None 表示不添加范数边界约束，非 None 时启用)
            norm_bound_type: 范数边界约束类型 "l1" 或 "linf" (仅 B_t 非 None 时生效)
                "l1": |π_j| ≤ B_t · w_j · π_0 (限制 π 的 L∞ 范数)
                "linf": Σ w_j |π_j| ≤ B_t · π_0 (限制 π 的加权 L1 范数)
            weights: 权重系数列表 (None 表示全为 1)
        """
        self.logger = logger
        self.dim_pi = dim_pi
        self.X_trial = X_trial
        self.theta_trial = theta_trial
        self.B_t = B_t
        self.norm_bound_type = norm_bound_type
        self.weights = weights
        self.outer_model = OuterModel(dim_pi, X_trial, theta_trial)

        # 添加范数边界约束 (B_t 非 None 时启用)
        if B_t is not None:
            self._add_norm_bound_constrains()

    def _add_norm_bound_constrains(self):
        """根据 norm_bound_type 添加范数边界约束"""
        if self.norm_bound_type == "l1":
            self.outer_model.add_l1_norm_bound_constrains(self.B_t, self.weights)
            self.logger.info(
                f"Added L1 norm bound constraints: |π_j| ≤ {self.B_t} · w_j · π_0"
            )
        elif self.norm_bound_type == "linf":
            self.outer_model.add_linf_norm_bound_constrains(self.B_t, self.weights)
            self.logger.info(
                f"Added L∞ norm bound constraints: Σ w_j |π_j| ≤ {self.B_t} · π_0"
            )
        else:
            raise ValueError(
                f"Unknown norm_bound_type: {self.norm_bound_type}, expected 'l1' or 'linf'"
            )

    def add_cut(self, subgradient):
        """
        添加切平面约束
        L <= pi^T * subgradient_pi + pi0 * subgradient_pi0

        Args:
            subgradient: pi 部分的次梯度 + pi0 部分的次梯度
        """
        self.outer_model.add_constrains(subgradient)

    def solve(self):
        """
        求解 outer model，返回 (pi_dummy, pi0_dummy, outer_obj)
        """
        self.outer_model.model.optimize()

        if self.outer_model.model.status != 2:
            self.outer_model.model.write("outer_model.lp")
            self.logger.warning(
                f"Outer model status: {self.outer_model.model.status}"
            )
            return None, None, None

        pi_dummy = [self.outer_model.pi[i].x for i in range(self.dim_pi)]
        pi0_dummy = self.outer_model.pi0.x
        outer_obj = self.outer_model.model.getObjective().getValue()

        return pi_dummy, pi0_dummy, outer_obj

    def set_level(self, level, pi_hat, pi0_hat):
        """
        设置 level 下界并切换为最小化与 (pi_hat, pi0_hat) 的距离
        """
        self.outer_model.set_lower_bound(level)
        self.outer_model.set_level_obj(pi_hat, pi0_hat)

    def recover(self):
        """恢复 outer model 为最大化模式"""
        self.outer_model.recover()


class LevelBundleSolver:
    """
    Level Bundle 算法求解器

    算法流程:
        1. 初始化 pi_hat = 0, pi0_hat = 1
        2. 求解 inner_model 得到次梯度，添加到 outer_model
        3. 求解 outer_model 得到 UB
        4. 计算 LB = inner_obj - pi_hat^T * X_trial - pi0_hat * theta_trial
        5. 判断收敛: UB - LB < gap_tol * UB
        6. Level 策略: level = UB - level_factor * (UB - LB)
        7. 在 outer_model 中设定 level 下界，最小化与当前中心的距离
        8. 求解得到新的 (pi_hat, pi0_hat)，回到步骤 2

    对偶边界参数 (通过构造器传入，均使用默认值时退化为基础算法):
        B_t: 对偶边界值，非 None 时在 outer model 中添加范数边界约束
        norm_bound_type: 范数边界约束类型 ("l1" 或 "linf")
        weights: 范数边界约束的权重系数列表
    """

    def __init__(self, logger, config, n, i_override=None,
                 B_t=None, norm_bound_type="l1", weights=None):
        """
        Args:
            logger: 日志器
            config: LevelBundleConfig 对象
            n: realization 索引
            i_override: 可选，覆盖 config 中的 iteration
            B_t: 对偶边界值 (默认 None，即不添加范数边界约束)
            norm_bound_type: 范数边界约束类型 "l1" 或 "linf" (默认 "l1")
            weights: 权重系数列表 (默认 None，即全为 1)
        """
        self.logger = logger
        self.config = config
        self.n = n
        self.i = i_override if i_override is not None else config.iteration

        # 初始化 inner problem
        self.inner_problem = InnerProblem(
            logger, config, n, i_override,
        )

        # 初始化 outer problem (B_t 非 None 时启用范数边界约束)
        self.outer_problem = OuterProblem(
            logger,
            dim_pi=config.N_VARS,
            X_trial=config.X_trial,
            theta_trial=config.THETA_TRIAL,
            B_t=B_t,
            norm_bound_type=norm_bound_type,
            weights=weights,
        )

    def solve(self) -> SolverResults:
        """
        执行 Level Bundle 算法

        Returns:
            SolverResults: 包含 pi_star, pi0_star, converged, lb, ub, n_iterations
        """
        from time import time

        start_time = time()

        config = self.config
        X_trial = config.X_trial
        theta_trial = config.THETA_TRIAL

        # 初始化乘子
        pi_hat = np.zeros(len(X_trial))
        pi0_hat = 0.1

        # 最优乘子
        pi_star = None
        pi0_star = None

        LB = float('-inf')
        UB = float('inf')
        subgradient_list = []
        lb_history = []
        ub_history = []
        time_history = []

        for iter_idx in range(config.iteration_limit):
            iter_start = time()

            # ============================
            # Step 1: 求解 inner model
            # ============================
            z_X_values, obj_term_value, inner_obj = self.inner_problem.solve(
                pi_hat, pi0_hat
            )

            if z_X_values is None:
                self.logger.warning(f"Iter {iter_idx}: Inner model failed, stopping")
                break

            # 构造次梯度
            subgradient = z_X_values + [obj_term_value]
            subgradient_list.append(subgradient)

            # ============================
            # Step 2: 添加切平面到 outer model
            # ============================
            self.outer_problem.add_cut(subgradient)

            # ============================
            # Step 3: 求解 outer model (最大化模式)
            # ============================
            pi_dummy, pi0_dummy, outer_obj = self.outer_problem.solve()

            if outer_obj is None:
                self.logger.warning(f"Iter {iter_idx}: Outer model failed, stopping")
                break

            # ============================
            # Step 4: 更新 LB 和 UB
            # ============================
            gap = inner_obj - sum(pi_hat[i] * X_trial[i] for i in range(len(X_trial))) - pi0_hat * theta_trial
            if gap > LB:
                LB = gap
                pi_star = pi_hat.copy()
                pi0_star = pi0_hat

            UB = outer_obj

            # 记录历史
            lb_history.append(LB)
            ub_history.append(UB)
            time_history.append(time() - iter_start)

            # ============================
            # Step 5: 判断收敛
            # ============================
            if UB - LB < config.gap_tol * abs(UB) or UB - LB < 1e-6:
                self.logger.info(
                    f"Level Bundle converged at iter {iter_idx}, "
                    f"LB: {LB:.6f}, UB: {UB:.6f}, gap: {UB - LB:.6e}"
                )
                if pi0_star > 1e-6 and LB / pi0_star >= config.pi0_tol * (abs(theta_trial) + 1):
                    elapsed = time() - start_time
                    results = SolverResults()
                    results.set_values(pi_star, pi0_star, True, LB, UB, iter_idx + 1, elapsed,
                                       lb_history, ub_history, time_history)
                    return results
                else:
                    # gap 已收敛但 pi0_star 过小或 LB/pi0_star 过小，
                    # 无法生成有效 Lagrangian cut，退出循环
                    self.logger.info(
                        f"pi0_star={pi0_star:.6f} too small or LB/pi0_star too small, "
                        f"no valid Lagrangian cut can be generated"
                    )
                    pi_star = None
                    pi0_star = None
                    break

            # ============================
            # Step 6: Level 策略
            # ============================
            level = UB - config.level_factor * (UB - LB)

            # 设置 level 下界并切换为最小化距离
            self.outer_problem.set_level(level, pi_hat, pi0_hat)

            self.outer_problem.outer_model.model.params.Method = 2
            self.outer_problem.outer_model.model.update()
            self.outer_problem.outer_model.model.optimize()

            if self.outer_problem.outer_model.model.status != 2:
                # 尝试 Method = 1
                self.outer_problem.outer_model.model.params.Method = 1
                self.outer_problem.outer_model.model.update()
                self.outer_problem.outer_model.model.optimize()

                if self.outer_problem.outer_model.model.status != 2:
                    # 尝试 Method = 0
                    self.outer_problem.outer_model.model.params.Method = 0
                    self.outer_problem.outer_model.model.update()
                    self.outer_problem.outer_model.model.optimize()

                    if self.outer_problem.outer_model.model.status != 2:
                        self.logger.warning(
                            f"Iter {iter_idx}: QP failed with all methods, recovering"
                        )
                        self.outer_problem.recover()
                        # 使用 outer model 的解作为新的 pi_hat
                        pi_hat = np.array(pi_dummy)
                        pi0_hat = pi0_dummy
                        continue

            # 获取新的乘子
            pi_hat = np.array([
                self.outer_problem.outer_model.pi[i].x
                for i in range(config.N_VARS)
            ])
            pi0_hat = self.outer_problem.outer_model.pi0.x

            # 恢复 outer model 为最大化模式
            self.outer_problem.recover()

            # 日志输出
            self.logger.info(
                f"Iter {iter_idx}: LB={LB:.6f}, UB={UB:.6f}, "
                f"gap={UB - LB:.6e}, pi0_hat={pi0_hat:.6f}"
            )

            # 时间限制检查
            if time() - start_time >= config.time_limit:
                self.logger.info(f"Time limit reached at iter {iter_idx}")
                break

        elapsed = time() - start_time
        results = SolverResults()
        results.set_values(pi_star, pi0_star, False, LB, UB, iter_idx + 1, elapsed,
                           lb_history, ub_history, time_history)

        return results


# def main():
#     from bundle_norm_RL.script.config import get_default_level_bundle_config
#     from bundle_norm_RL.script.logger import get_logger
#
#     log = get_logger("./level_bundle_main.log")
#
#     config = get_default_level_bundle_config()
#     solver = LevelBundleSolver(log, config, n=0)
#
#     results = solver.solve()
#
#     log.info(f"Results: {results.toString()}")
#
#     # 如果收敛且 pi0_star 有效，计算最终的 Lagrangian cut
#     # (pi0_star 为 None 表示 gap 收敛但无法生成有效 cut)
#     if results.converged and results.pi0_star is not None and results.pi0_star > 1e-6:
#         pi_star = results.pi_star
#         pi0_star = results.pi0_star
#
#         # 重新求解 inner model 获取最终 cut
#         inner = InnerProblem(log, config, n=0)
#         inner.model.setParam("OutputFlag", 0)
#         z_X, obj_term = inner.uc_bw.add_inner_objective(
#             config.PROBLEM_PARAMS.cost_coeffs, pi_star, pi0_star
#         )
#         inner.model.optimize()
#
#         intercept = inner.model.getObjective().getValue()
#         pi = -pi_star / pi0_star
#         intercept = intercept / pi0_star
#
#         log.info(f"Lagrangian cut gradient (pi): {pi}")
#         log.info(f"Lagrangian cut intercept: {intercept}")
#
#
# def main_norm():
#     """
#     带对偶边界约束的 Level Bundle 主函数示例
#
#     对偶边界参数通过 LevelBundleSolver 的构造器传入:
#         B_t: 对偶边界值 (默认 None，即不添加范数边界约束)
#         norm_bound_type: 范数边界约束类型 "l1" 或 "linf" (默认 "l1")
#         weights: 权重系数列表 (默认 None，即全为 1)
#     """
#     from bundle_norm_RL.script.config import get_default_level_bundle_config
#     from bundle_norm_RL.script.logger import get_logger
#
#     log = get_logger("./level_bundle_norm_main.log")
#
#     config = get_default_level_bundle_config()
#     solver = LevelBundleSolver(
#         log, config, n=0,
#         B_t=1.0,            # 对偶边界值
#         norm_bound_type="l1",  # 范数边界约束类型
#     )
#
#     results = solver.solve()
#
#     log.info(f"Results: {results.toString()}")
#
#     # 如果收敛且 pi0_star 有效，计算最终的 Lagrangian cut
#     # (pi0_star 为 None 表示 gap 收敛但无法生成有效 cut)
#     if results.converged and results.pi0_star is not None and results.pi0_star > 1e-6:
#         pi_star = results.pi_star
#         pi0_star = results.pi0_star
#
#         # 重新求解 inner model 获取最终 cut
#         inner = InnerProblem(log, config, n=0)
#         inner.model.setParam("OutputFlag", 0)
#         z_X, obj_term = inner.uc_bw.add_inner_objective(
#             config.PROBLEM_PARAMS.cost_coeffs, pi_star, pi0_star
#         )
#         inner.model.optimize()
#
#         intercept = inner.model.getObjective().getValue()
#         pi = -pi_star / pi0_star
#         intercept = intercept / pi0_star
#
#         log.info(f"Lagrangian cut gradient (pi): {pi}")
#         log.info(f"Lagrangian cut intercept: {intercept}")


def load_config_and_solve(
    log=None,
    configs_dir: str = r"D:\tools\workspace_pycharm\SDDiP-RL\bundle_norm_RL\configs",
    # 对偶边界参数 (均使用默认值时退化为基础算法)
    B_t: float = None,
    norm_bound_type: str = "l1",
    weights: list = None,
) -> SolverResults:
    """
    从保存的 pkl 文件加载 LevelBundleConfig 并求解

    Args:
        log: 日志器
        configs_dir: config pkl 文件所在目录
        B_t: 对偶边界值 (None 则不添加范数边界约束)
        norm_bound_type: 范数边界约束类型 ("l1" 或 "linf")
        weights: 权重系数列表

    Returns:
        SolverResults
    """
    from bundle_norm_RL.script.config import LevelBundleConfig

    ratio_list = []
    # for i in range(2, 3):
    #     count = 0
    #     all = 0
    #     for t in range(1, 24):
    #         for n in range(0, 6):
    i = 2
    t = 2
    n = 0
    pkl_path = f"{configs_dir}/config_{i}_{t}_{n}.pkl"
    config = LevelBundleConfig.from_pkl(pkl_path)

    log.info(f"Loaded config from {pkl_path}")
    log.info(config.toString())

    solver = LevelBundleSolver(
        log, config, n=n,
        B_t=B_t, norm_bound_type=norm_bound_type,
        weights=weights,
    )
    results = solver.solve()

    log.info(f"Results: {results.toString()}")
    # if results.converged:
    #     count += 1
    # all += 1
    # ratio_list.append(count / all)
    # print(f"ratio: {[f'{r:.5f}' for r in ratio_list]}")


if __name__ == "__main__":
    # main()
    # main_norm()
    from bundle_norm_RL.script.logger import get_logger
    log = get_logger("../logs/level_bundle_norm.log")
    load_config_and_solve(log)
