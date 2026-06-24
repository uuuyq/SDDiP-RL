"""
Level Bundle 算法求解脚本 (带正则化策略与范数边界约束)

在基础 Level Bundle 算法上新增:
    1. 正则化策略: 在 inner objective 中添加 σ · ||z_X - X_trial|| 项
       - L1 范数正则化: σ · Σ_j |z_X_j - X_trial_j|
       - L∞ 范数正则化: σ · max_j |z_X_j - X_trial_j|
    2. 范数边界约束: 在 outer model 中添加 |π_j| ≤ B_t · w_j · π_0 或 Σ w_j |π_j| ≤ B_t · π_0
       - L1 范数边界: 限制 π 的 L∞ 范数
       - L∞ 范数边界: 限制 π 的加权 L1 范数

符号说明:
    sigma_t: 正则化系数 σ_t
    B_t: 对偶边界值 (默认 B_t = σ_t)
    norm_type: 正则化范数类型 ("l1" 或 "linf")
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

    def set_values(self, pi_star, pi0_star, converged, lb, ub, n_iterations, solver_time):
        self.pi_star = pi_star
        self.pi0_star = pi0_star
        self.converged = converged
        self.lb = lb
        self.ub = ub
        self.n_iterations = n_iterations
        self.solver_time = solver_time

    def toString(self):
        return (
            f"pi_star: {self.pi_star}, pi0_star: {self.pi0_star:.6f}, "
            f"converged: {self.converged}, LB: {self.lb:.4f}, UB: {self.ub:.4f}, "
            f"iterations: {self.n_iterations}, time: {self.solver_time:.4f}s"
        )


class InnerProblemNorm:
    """
    Inner Problem (带正则化): 给定乘子 (pi_hat, pi0_hat)，求解 Lagrangian 松弛子问题

    min_{x,z} pi_hat^T * z_X + pi0_hat * obj_term + sigma * ||z_X - X_trial||

    其中 z_X = (z_x, z_y, z_x_bs, z_soc) 为 copy 变量
    """

    def __init__(self, logger, config, n, i_override=None):
        """
        Args:
            logger: 日志器
            config: LevelBundleConfigNorm 对象
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
        # 正则化参数
        self.sigma = config.sigma_t
        self.norm_type = config.norm_type
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
        给定乘子 (pi_hat, pi0_hat)，设置带正则化的 inner objective 并求解

        Returns:
            z_X_values: z 变量的取值 (次梯度的 pi 部分)
            obj_term_value: theta 对应项的取值 (次梯度的 pi0 部分)
            inner_obj: inner model 的目标函数值 (含正则化项)
            reg_offset: 正则化项的值 σ · ||z_X_values - X_trial|| (用于 cut 偏移)
        """
        if self.sigma > 0:
            z_X, obj_term = self.uc_bw.add_inner_objective_with_regularization(
                self.problem_params.cost_coeffs, pi_hat, pi0_hat,
                sigma=self.sigma, X_trial=self.X_trial, norm_type=self.norm_type
            )
        else:
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
            return None, None, None, None

        z_X_values = [z_X[i].x for i in range(len(z_X))]
        obj_term_value = obj_term.getValue()
        inner_obj = self.model.getObjective().getValue()

        # 计算正则化项的值 σ · ||z_X_values - X_trial||
        reg_offset = self._compute_reg_offset(z_X_values)

        return z_X_values, obj_term_value, inner_obj, reg_offset

    def _compute_reg_offset(self, z_X_values):
        """
        计算正则化项的值 σ · ||z_X_values - X_trial||

        用于在 outer model 中添加带偏移的 cut:
            L ≤ z_X^T · π + obj_term · π0 + σ · ||z_X_values - X_trial||
        """
        if self.sigma <= 0:
            return 0.0

        diff = np.array(z_X_values) - np.array(self.X_trial)

        if self.norm_type == "l1":
            return self.sigma * np.sum(np.abs(diff))
        elif self.norm_type == "linf":
            return self.sigma * np.max(np.abs(diff))
        else:
            raise ValueError(f"Unknown norm_type: {self.norm_type}")


class OuterProblemNorm:
    """
    Outer Problem (带范数边界约束): 收集次梯度，构造切平面，求解得到新的乘子

    最大化: L - pi^T * X_trial - pi0 * theta_trial
    约束:
        L <= pi^T * subgradient_pi + pi0 * subgradient_pi0 + reg_offset  (切平面)
        ||pi||_1 + pi0 <= 1  (归一化约束)
        |π_j| ≤ B_t · w_j · π_0  (L1 范数边界约束, 限制 L∞)
        或 Σ w_j |π_j| ≤ B_t · π_0  (L∞ 范数边界约束, 限制加权 L1)
    """

    def __init__(self, logger, dim_pi, X_trial, theta_trial,
                 B_t=None, norm_bound_type="l1", weights=None):
        """
        Args:
            logger: 日志器
            dim_pi: pi 的维度
            X_trial: trial point (拼接后)
            theta_trial: theta 的 trial point
            B_t: 对偶边界值 (None 表示不添加范数边界约束)
            norm_bound_type: "l1" 或 "linf"
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

        # 添加范数边界约束
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

    def add_cut(self, subgradient, reg_offset=0.0):
        """
        添加切平面约束 (带正则化偏移)
        L <= pi^T * subgradient_pi + pi0 * subgradient_pi0 + reg_offset

        Args:
            subgradient: pi 部分的次梯度 + pi0 部分的次梯度
            reg_offset: 正则化项的值 (σ · ||z_X_values - X_trial||)
        """
        if reg_offset > 0:
            self.outer_model.add_constrains_with_offset(subgradient, reg_offset)
        else:
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


class LevelBundleSolverNorm:
    """
    Level Bundle 算法求解器 (带正则化策略与范数边界约束)

    算法流程:
        1. 初始化 pi_hat = 0, pi0_hat = 1
        2. 求解带正则化的 inner_model 得到次梯度，计算 reg_offset
        3. 将带偏移的切平面添加到 outer_model (含范数边界约束)
        4. 求解 outer_model 得到 UB
        5. 计算 LB = inner_obj - pi_hat^T * X_trial - pi0_hat * theta_trial
        6. 判断收敛: UB - LB < gap_tol * UB
        7. Level 策略: level = UB - level_factor * (UB - LB)
        8. 在 outer_model 中设定 level 下界，最小化与当前中心的距离
        9. 求解得到新的 (pi_hat, pi0_hat)，回到步骤 2
    """

    def __init__(self, logger, config, n, i_override=None):
        """
        Args:
            logger: 日志器
            config: LevelBundleConfigNorm 对象
            n: realization 索引
            i_override: 可选，覆盖 config 中的 iteration
        """
        self.logger = logger
        self.config = config
        self.n = n
        self.i = i_override if i_override is not None else config.iteration

        # 初始化 inner problem (带正则化)
        self.inner_problem = InnerProblemNorm(logger, config, n, i_override)

        # 初始化 outer problem (带范数边界约束)
        self.outer_problem = OuterProblemNorm(
            logger,
            dim_pi=config.N_VARS,
            X_trial=config.X_trial,
            theta_trial=config.THETA_TRIAL,
            B_t=config.B_t,
            norm_bound_type=config.norm_bound_type,
            weights=config.weights,
        )

    def solve(self) -> SolverResults:
        """
        执行 Level Bundle 算法 (带正则化与范数边界约束)

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
        pi0_hat = 0.001

        # 最优乘子
        pi_star = None
        pi0_star = None

        LB = float('-inf')
        UB = float('inf')
        subgradient_list = []

        for iter_idx in range(config.iteration_limit):
            # ============================
            # Step 1: 求解 inner model (带正则化)
            # ============================
            z_X_values, obj_term_value, inner_obj, reg_offset = self.inner_problem.solve(
                pi_hat, pi0_hat
            )

            if z_X_values is None:
                self.logger.warning(f"Iter {iter_idx}: Inner model failed, stopping")
                break

            # 构造次梯度
            subgradient = z_X_values + [obj_term_value]
            subgradient_list.append(subgradient)

            # ============================
            # Step 2: 添加带偏移的切平面到 outer model
            # ============================
            self.outer_problem.add_cut(subgradient, reg_offset=reg_offset)

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
                    results.set_values(pi_star, pi0_star, True, LB, UB, iter_idx + 1, elapsed)
                    return results

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
                f"gap={UB - LB:.6e}, pi0_hat={pi0_hat:.6f}, "
                f"reg_offset={reg_offset:.6f}"
            )

            # 时间限制检查
            if time() - start_time >= config.time_limit:
                self.logger.info(f"Time limit reached at iter {iter_idx}")
                break

        elapsed = time() - start_time
        results = SolverResults()
        results.set_values(pi_star, pi0_star, False, LB, UB, iter_idx + 1, elapsed)

        return results


def main():
    from bundle_norm_RL.script.config import get_default_level_bundle_config_norm
    from bundle_norm_RL.script.logger import get_logger

    log = get_logger("./level_bundle_norm_main.log")

    config = get_default_level_bundle_config_norm()
    solver = LevelBundleSolverNorm(log, config, n=0)

    results = solver.solve()

    log.info(f"Results: {results.toString()}")

    # 如果收敛，计算最终的 Lagrangian cut
    if results.converged and results.pi0_star > 1e-6:
        pi_star = results.pi_star
        pi0_star = results.pi0_star

        # 重新求解 inner model 获取最终 cut
        inner = InnerProblemNorm(log, config, n=0)
        inner.model.setParam("OutputFlag", 0)
        z_X, obj_term = inner.uc_bw.add_inner_objective_with_regularization(
            config.PROBLEM_PARAMS.cost_coeffs, pi_star, pi0_star,
            sigma=config.sigma_t, X_trial=config.X_trial, norm_type=config.norm_type
        )
        inner.model.optimize()

        intercept = inner.model.getObjective().getValue()
        pi = -pi_star / pi0_star
        intercept = intercept / pi0_star

        log.info(f"Lagrangian cut gradient (pi): {pi}")
        log.info(f"Lagrangian cut intercept: {intercept}")


def load_config_and_solve(
    log=None,
    configs_dir: str = r"D:\tools\workspace_pycharm\SDDiP-RL\bundle_norm_RL\configs",
    sigma_t: float = 10.0,
    B_t: float = None,
    norm_type: str = "l1",
    norm_bound_type: str = "l1",
) -> SolverResults:
    """
    从保存的 pkl 文件加载 LevelBundleConfig 并求解 (带正则化与范数边界约束)

    Args:
        log: 日志器
        configs_dir: config pkl 文件所在目录
        sigma_t: 正则化系数
        B_t: 对偶边界值 (None 则默认等于 sigma_t)
        norm_type: 正则化范数类型 ("l1" 或 "linf")
        norm_bound_type: 范数边界约束类型 ("l1" 或 "linf")

    Returns:
        SolverResults
    """
    from bundle_norm_RL.script.config import LevelBundleConfig, LevelBundleConfigNorm
    count = 0
    all = 0
    if B_t is None:
        B_t = sigma_t

    for i in range(8, 11):
        for t in range(2, 12):
            for n in range(0, 1):

                pkl_path = f"{configs_dir}/config_{i}_{t}_{n}.pkl"
                base_config = LevelBundleConfig.from_pkl(pkl_path)

                print(base_config.toString())

                # 转换为带正则化的配置
                config = LevelBundleConfigNorm.from_base_config(
                    base_config,
                    sigma_t=sigma_t,
                    B_t=B_t,
                    norm_type=norm_type,
                    norm_bound_type=norm_bound_type,
                )

                log.info(f"Loaded config from {pkl_path}")
                log.info(config.toString())

                solver = LevelBundleSolverNorm(log, config, n=n)
                results = solver.solve()

                log.info(f"Results: {results.toString()}")
                if results.converged:
                    count += 1
                all += 1
    print(f"ratio: {count}/{all}")


if __name__ == "__main__":
    # main()
    from bundle_norm_RL.script.logger import get_logger
    log_path = "level_bundle_norm_loaded.log"
    log = get_logger(log_path)

    load_config_and_solve(log)
