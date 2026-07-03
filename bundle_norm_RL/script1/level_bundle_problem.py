"""
Level Bundle 算法求解脚本 —— 增量重参数化形式

基于 increment.tex 文档中的增量重参数化推导:

    Master Problem 的增量形式:
        max_{d, L}  L - (ρ/2) ||d||²
        s.t.        L ≤ (g^i)^T d + β_i,  ∀ i ∈ I_t

    其中:
        d = λ - λ̄ = [d_π; d_{π0}]  为增量变量
        λ̄ = [π̄; π̄₀]               为当前稳定中心 (Serious Center)
        g^i = [g_π^i; g_{π0}^i]    为第 i 个 bundle cut 对应的次梯度
        β_i = (g^i)^T(λ̄ - λ^i) + ω(λ^i)  为常数项

    恢复原始对偶变量: λ = λ̄ + d

    KKT 条件给出:
        d* = (1/ρ) Σ μ_i g^i
        Σ μ_i = 1

符号说明:
    pi_bar: 稳定中心 pi 部分 (对应 λ̄ 的 π 部分)
    pi0_bar: 稳定中心 pi0 部分 (对应 λ̄ 的 π₀ 部分)
    d_pi: 增量 pi 部分 (d_π = π - π̄)
    d_pi0: 增量 pi0 部分 (d_{π0} = π₀ - π̄₀)
    rho: proximal 惩罚参数
    omega_i: 第 i 个 cut 在试探点 λ^i 处的函数值 (即 inner_obj_i)
"""

import logging

import numpy as np
import gurobipy as gp

from sddip.sddip import ucmodelclassical
from sddip.sddip.outermodel import OuterModel


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


class IncrementalOuterProblem:
    """
    增量形式的 Outer Problem

    Master Problem 使用两种模式交替求解:

    1. 最大化模式 (求 UB):
        max_{d, L}  L - π^T·X - π₀·θ
        s.t.        L ≤ (g^i)^T d + β_i,  ∀ i ∈ I_t
                    Σ|π_j| + π₀ ≤ 1  (归一化约束)
                    π₀ ≥ 1e-4
                    (可选) 范数边界约束

       其中 π = π̄ + d_π, π₀ = π̄₀ + d_{π0}
       UB = L* - π*^T·X - π₀*·θ (对偶函数上界)

    2. Level 模式 (求试探点):
        min_{d}  (ρ/2) ||d||²
        s.t.     L - π^T·X - π₀·θ ≥ level
                 (所有 cut 和归一化约束)

       proximal 项 (ρ/2)||d||² 限制增量大小，使试探点不偏离稳定中心太远

    切平面约束 (增量形式):
        L ≤ (g^i)^T d + β_i
        其中:
            d = λ - λ̄ = [d_π; d_{π0}]  为增量变量
            λ̄ = [π̄; π̄₀]               为当前稳定中心 (Serious Center)
            g^i = [g_π^i; g_{π0}^i]    为第 i 个 bundle cut 对应的次梯度
            β_i = (g^i)^T(λ̄ - λ^i) + ω(λ^i)  为常数项

    恢复原始对偶变量: λ = λ̄ + d

    对偶边界约束扩展 (通过参数控制是否开启):
        范数边界约束: 在 outer model 中添加 |π_j| ≤ B_t · w_j · π_0 或 Σ w_j |π_j| ≤ B_t · π_0
    """

    def __init__(self, logger, dim_pi, X_trial, theta_trial,
                 rho=1.0, B_t=None, norm_bound_type="l1", weights=None):
        """
        Args:
            logger: 日志器
            dim_pi: pi 的维度
            X_trial: trial point (拼接后)
            theta_trial: theta 的 trial point
            rho: proximal 惩罚参数 (默认 1.0)
            B_t: 对偶边界值 (None 表示不添加范数边界约束，非 None 时启用)
            norm_bound_type: 范数边界约束类型 "l1" 或 "linf" (仅 B_t 非 None 时生效)
            weights: 权重系数列表 (None 表示全为 1)
        """
        self.logger = logger
        self.dim_pi = dim_pi
        self.X_trial = X_trial
        self.theta_trial = theta_trial
        self.rho = rho
        self.B_t = B_t
        self.norm_bound_type = norm_bound_type
        self.weights = weights

        # 稳定中心 (Serious Center)
        self.pi_bar = np.zeros(dim_pi)
        self.pi0_bar = 0.1

        # Bundle 信息: 存储每个 cut 的 (lambda^i, g^i, omega^i)
        # lambda^i: 生成第 i 个 cut 时的乘子 [π^i; π₀^i]
        # g^i: 次梯度 [g_π; g_{π0}]
        # omega^i: 在 λ^i 处的函数值 (inner_obj - π^i^T·X_trial - π₀^i·θ_trial)
        self.bundle_info = []

        # 构建 Gurobi 模型
        self.model = gp.Model("incremental_outer_model")
        self.model.setParam("OutputFlag", 0)

        # 增量变量 d = [d_π; d_{π0}]
        self.d_pi = []
        self.d_pi0 = None
        self.L = None

        # 辅助变量 (π, π₀) 用于 abs 约束和范数边界约束
        self.pi_vars = []
        self.pi0_var = None
        self.abs_pi = []

        # 目标函数表达式
        self.max_obj = None      # 最大化模式: L - π^T·X - π₀·θ
        self.proximal_obj = None  # Level 模式: (ρ/2)||d||²

        self.lower_bound_const = None

        self._init_model()

        # 添加范数边界约束 (B_t 非 None 时启用)
        if B_t is not None:
            self._add_norm_bound_constrains()

    def _init_model(self):
        """初始化增量形式的 Master Problem"""
        # 增量变量 d_π (无界)
        for i in range(self.dim_pi):
            self.d_pi.append(
                self.model.addVar(
                    vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY,
                    name="d_pi_%i" % (i + 1)
                )
            )
        # 增量变量 d_{π0} (无界)
        self.d_pi0 = self.model.addVar(
            vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="d_pi0"
        )

        # 辅助变量: π_j = π̄_j + d_π_j (用于 abs 约束和目标函数)
        self.pi_vars = []
        for i in range(self.dim_pi):
            self.pi_vars.append(
                self.model.addVar(
                    vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY,
                    name="pi_%i" % (i + 1)
                )
            )
        # 辅助变量: π₀ = π̄₀ + d_{π0} (下界 1e-4, 与原始 OuterModel 一致)
        self.pi0_var = self.model.addVar(
            vtype=gp.GRB.CONTINUOUS, lb=1e-4, name="pi0"
        )

        # 约束: π_j = π̄_j + d_π_j
        for i in range(self.dim_pi):
            self.model.addConstr(
                self.pi_vars[i] == self.pi_bar[i] + self.d_pi[i],
                name="pi_def_%i" % (i + 1)
            )
        # 约束: π₀ = π̄₀ + d_{π0}
        self.model.addConstr(
            self.pi0_var == self.pi0_bar + self.d_pi0,
            name="pi0_def"
        )

        # L1 归一化约束: Σ|π_j| + π₀ ≤ 1
        self.abs_pi = []
        for i in range(self.dim_pi):
            self.abs_pi.append(
                self.model.addVar(
                    vtype=gp.GRB.CONTINUOUS, lb=0, name="abs_pi_%i" % (i + 1)
                )
            )
        # 必须在 addConstr + gp.abs_ 之前 update，否则变量尚未注册
        self.model.update()
        for i in range(self.dim_pi):
            self.model.addConstr(
                self.abs_pi[i] == gp.abs_(self.pi_vars[i]),
                name="abs_pi_def_%i" % (i + 1)
            )
        self.model.addConstr(
            gp.quicksum(self.abs_pi[i] for i in range(self.dim_pi))
            + self.pi0_var <= 1,
            name="l1_norm"
        )

        # 辅助变量 L
        self.L = self.model.addVar(
            vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="L"
        )

        # 最大化模式目标函数: L - π^T·X - π₀·θ (对偶函数上界)
        self.max_obj = (
            self.L
            - gp.LinExpr(self.X_trial, self.pi_vars)
            - self.pi0_var * self.theta_trial
        )

        # Level 模式目标函数: (ρ/2)||d||²
        self.proximal_obj = (self.rho / 2) * (
            gp.quicksum(self.d_pi[i] ** 2 for i in range(self.dim_pi))
            + self.d_pi0 ** 2
        )

        # 默认使用最大化模式
        self.model.setObjective(self.max_obj, gp.GRB.MAXIMIZE)
        self.model.update()

    def set_stability_center(self, pi_bar, pi0_bar):
        """
        更新稳定中心 λ̄ = (π̄, π̄₀)

        当稳定中心更新时，需要:
        1. 更新 π = π̄ + d 的定义约束
        2. 重新计算所有已有 cut 的常数项 β_i
        3. 更新范数边界约束 (如有)
        """
        self.pi_bar = np.array(pi_bar, dtype=float)
        self.pi0_bar = float(pi0_bar)
        self._rebuild_model()

    def _rebuild_model(self):
        """重建模型 (稳定中心变化后调用)"""
        # 释放旧模型并创建新模型 (直接 remove vars 会破坏 abs 等通用约束)
        self.model.dispose()

        self.model = gp.Model("incremental_outer_model")
        self.model.setParam("OutputFlag", 0)

        self.d_pi = []
        self.d_pi0 = None
        self.pi_vars = []
        self.pi0_var = None
        self.L = None
        self.abs_pi = []
        self.max_obj = None
        self.proximal_obj = None
        self.norm_bound_constrs = []

        # 重新初始化模型
        self._init_model()

        # 重新添加所有 cut 约束 (常数项 β_i 依赖 λ̄)
        for info in self.bundle_info:
            self._add_cut_constr(info['lambda_i'], info['g_i'], info['omega_i'])

        # 重新添加范数边界约束
        if self.B_t is not None:
            self._add_norm_bound_constrains()

    def add_cut(self, subgradient, lambda_i, omega_i):
        """
        添加增量形式的切平面约束

        原始 cut: L ≤ ω(λ^i) + (g^i)^T(λ̄ + d - λ^i)
        整理后: L ≤ (g^i)^T d + β_i
        其中 β_i = (g^i)^T(λ̄ - λ^i) + ω(λ^i)

        Args:
            subgradient: 次梯度 [g_π; g_{π0}] (list 或 array)
            lambda_i: 生成该 cut 时的乘子 [π^i; π₀^i] (list 或 array)
            omega_i: 在 λ^i 处的函数值 (inner_obj - π^i^T·X_trial - π₀^i·θ_trial)
        """
        g_i = np.array(subgradient, dtype=float)
        lam_i = np.array(lambda_i, dtype=float)

        # 存储 bundle 信息
        self.bundle_info.append({
            'lambda_i': lam_i,
            'g_i': g_i,
            'omega_i': omega_i,
        })

        self._add_cut_constr(lam_i, g_i, omega_i)

    def _add_cut_constr(self, lam_i, g_i, omega_i):
        """
        添加单条切平面约束: L ≤ (g^i)^T d + β_i

        原始 cut: L ≤ (g^i)^T λ = (g^i)^T (λ̄ + d) = (g^i)^T d + (g^i)^T λ̄
        因此 β_i = (g^i)^T λ̄ (常数项，仅依赖稳定中心)

        注: omega_i 和 lam_i 保留用于兼容接口，但 β_i 的计算不依赖它们
        """
        # 分离 g_π 和 g_{π0}
        g_pi = g_i[:self.dim_pi]
        g_pi0 = g_i[self.dim_pi]

        # 计算常数项 β_i = (g^i)^T λ̄ = g_π^T π̄ + g_{π0} * π̄₀
        beta_i = g_pi @ self.pi_bar + g_pi0 * self.pi0_bar

        # 约束: L ≤ (g_π)^T d_π + g_{π0} * d_{π0} + β_i
        self.model.addConstr(
            self.L
            <= gp.LinExpr(g_pi, self.d_pi) + g_pi0 * self.d_pi0 + beta_i,
            name="cut_%i" % len(self.bundle_info)
        )
        self.model.update()

    def solve(self):
        """
        求解增量形式的 outer model (最大化模式)

        Returns:
            pi_dummy, pi0_dummy, ub
            其中 pi_dummy = π̄ + d_π*, pi0_dummy = π̄₀ + d_{π0}*
            ub = L* - π^T·X - π₀·θ (对偶函数上界)
        """
        self.model.optimize()

        if self.model.status != 2:
            self.model.write("incremental_outer_model.lp")
            self.logger.warning(
                f"Incremental outer model status: {self.model.status}"
            )
            return None, None, None

        # 恢复原始对偶变量: λ = λ̄ + d*
        d_pi_star = np.array([self.d_pi[i].x for i in range(self.dim_pi)])
        d_pi0_star = self.d_pi0.x

        pi_dummy = self.pi_bar + d_pi_star
        pi0_dummy = self.pi0_bar + d_pi0_star

        # UB = 对偶函数上界 (最大化模式的目标函数值)
        ub = self.model.getObjective().getValue()

        return pi_dummy.tolist(), pi0_dummy, ub

    def set_level(self, level):
        """
        设置 level 下界约束并切换为 Level 模式

        Level 模式: 在满足 dual_val ≥ level 的条件下，
        最小化与稳定中心的距离 (proximal step)

            min (ρ/2)‖d‖²
            s.t. L - π^T·X - π₀·θ ≥ level
        """
        # Level 下界约束: 对偶函数值 ≥ level
        self.lower_bound_const = self.model.addConstr(
            self.max_obj >= level,
            name="level_bound"
        )
        # 切换目标为最小化 ‖d‖² (proximal step)
        self.model.setObjective(self.proximal_obj, gp.GRB.MINIMIZE)
        self.model.update()

    def recover(self):
        """恢复 outer model 为最大化模式 (移除 level 下界约束，恢复目标函数)"""
        try:
            self.model.remove(self.lower_bound_const)
        except Exception as e:
            self.logger.debug(f"recover: {e}")
        # 恢复目标函数为最大化 L - π^T·X - π₀·θ
        self.model.setObjective(self.max_obj, gp.GRB.MAXIMIZE)
        self.model.update()

    def _add_norm_bound_constrains(self):
        """根据 norm_bound_type 添加范数边界约束"""
        if self.norm_bound_type == "l1":
            self._add_l1_norm_bound_constrains()
            self.logger.info(
                f"Added L1 norm bound constraints: |π_j| ≤ {self.B_t} · w_j · π_0"
            )
        elif self.norm_bound_type == "linf":
            self._add_linf_norm_bound_constrains()
            self.logger.info(
                f"Added L∞ norm bound constraints: Σ w_j |π_j| ≤ {self.B_t} · π_0"
            )
        else:
            raise ValueError(
                f"Unknown norm_bound_type: {self.norm_bound_type}, expected 'l1' or 'linf'"
            )

    def _add_l1_norm_bound_constrains(self):
        """
        L1 范数边界约束: |π_j| ≤ B_t · w_j · π_0

        等价于: ||π||_∞ ≤ B_t · ||w||_∞ · π_0

        其中 π_j = π̄_j + d_π_j, π₀ = π̄₀ + d_{π0}
        """
        if self.weights is None:
            weights = [1.0] * self.dim_pi
        else:
            weights = self.weights

        if not self.abs_pi:
            raise RuntimeError("abs_pi variables not initialized.")

        self.norm_bound_constrs = []
        for j in range(self.dim_pi):
            constr = self.model.addConstr(
                self.abs_pi[j] <= self.B_t * weights[j] * self.pi0_var,
                name=f"l1_norm_bound_{j + 1}"
            )
            self.norm_bound_constrs.append(constr)
        self.model.update()

    def _add_linf_norm_bound_constrains(self):
        """
        L∞ 范数边界约束: Σ w_j |π_j| ≤ B_t · π_0

        等价于: ||π||_{w,1} ≤ B_t · π_0

        其中 π_j = π̄_j + d_π_j, π₀ = π̄₀ + d_{π0}
        """
        if self.weights is None:
            weights = [1.0] * self.dim_pi
        else:
            weights = self.weights

        if not self.abs_pi:
            raise RuntimeError("abs_pi variables not initialized.")

        weighted_abs_pi = gp.quicksum(
            weights[j] * self.abs_pi[j] for j in range(self.dim_pi)
        )
        self.norm_bound_constrs = [
            self.model.addConstr(
                weighted_abs_pi <= self.B_t * self.pi0_var,
                name="linf_norm_bound"
            )
        ]
        self.model.update()


class LevelBundleSolver:
    """
    Level Bundle 算法求解器 —— 增量重参数化形式

    算法流程:
        1. 初始化 π̄ = 0, π̄₀ = 0.1 (稳定中心)
        2. 求解 inner_model 得到次梯度 g^i 和函数值 ω(λ^i)
        3. 将 (λ^i, g^i, ω(λ^i)) 添加到增量形式的 outer_model
        4. 求解 outer_model 得到 d* 和 UB
        5. 恢复 λ = λ̄ + d*, 计算 LB
        6. 判断收敛: UB - LB < gap_tol * |UB|
        7. Level 策略: level = UB - level_factor * (UB - LB)
        8. 在 outer_model 中设定 level 下界
        9. 求解得到新的 d*, 恢复新的 λ, 回到步骤 2
        10. Serious step: 更新稳定中心 λ̄ ← λ

    对偶边界参数 (通过构造器传入，均使用默认值时退化为基础算法):
        B_t: 对偶边界值，非 None 时在 outer model 中添加范数边界约束
        norm_bound_type: 范数边界约束类型 ("l1" 或 "linf")
        weights: 范数边界约束的权重系数列表
    """

    def __init__(self, logger, config, n, i_override=None,
                 rho=1.0, B_t=None, norm_bound_type="l1", weights=None):
        """
        Args:
            logger: 日志器
            config: LevelBundleConfig 对象
            n: realization 索引
            i_override: 可选，覆盖 config 中的 iteration
            rho: proximal 惩罚参数 (默认 1.0)
            B_t: 对偶边界值 (默认 None，即不添加范数边界约束)
            norm_bound_type: 范数边界约束类型 "l1" 或 "linf" (默认 "l1")
            weights: 权重系数列表 (默认 None，即全为 1)
        """
        self.logger = logger
        self.config = config
        self.n = n
        self.i = i_override if i_override is not None else config.iteration
        self.rho = rho

        # 初始化 inner problem
        self.inner_problem = InnerProblem(
            logger, config, n, i_override,
        )

        # 初始化增量形式的 outer problem
        self.outer_problem = IncrementalOuterProblem(
            logger,
            dim_pi=config.N_VARS,
            X_trial=config.X_trial,
            theta_trial=config.THETA_TRIAL,
            rho=rho,
            B_t=B_t,
            norm_bound_type=norm_bound_type,
            weights=weights,
        )

    def solve(self) -> SolverResults:
        """
        执行 Level Bundle 算法 (增量形式)

        Returns:
            SolverResults: 包含 pi_star, pi0_star, converged, lb, ub, n_iterations
        """
        from time import time

        start_time = time()

        config = self.config
        X_trial = config.X_trial
        theta_trial = config.THETA_TRIAL

        # 初始化乘子和稳定中心
        pi_hat = np.zeros(len(X_trial))
        pi0_hat = 0.1
        pi_bar = np.zeros(len(X_trial))
        pi0_bar = 0.1

        # 最优乘子
        pi_star = None
        pi0_star = None

        LB = float('-inf')
        UB = float('inf')
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

            # 构造次梯度 g^i = [g_π; g_{π0}] = [z^i - x; f^i + θ^i - θ]
            # 在当前代码中, subgradient = z_X_values + [obj_term_value]
            subgradient = z_X_values + [obj_term_value]

            # 计算 ω(λ^i) = inner_obj - π^i^T·X_trial - π₀^i·θ_trial
            lambda_i = np.concatenate([pi_hat, [pi0_hat]])
            omega_i = inner_obj - pi_hat @ X_trial - pi0_hat * theta_trial

            # ============================
            # Step 2: 添加切平面到增量 outer model
            # ============================
            self.outer_problem.add_cut(subgradient, lambda_i, omega_i)

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

            # 设置 level 下界
            self.outer_problem.set_level(level)

            self.outer_problem.model.params.Method = 2
            self.outer_problem.model.update()
            self.outer_problem.model.optimize()

            if self.outer_problem.model.status != 2:
                # 尝试 Method = 1
                self.outer_problem.model.params.Method = 1
                self.outer_problem.model.update()
                self.outer_problem.model.optimize()

                if self.outer_problem.model.status != 2:
                    # 尝试 Method = 0
                    self.outer_problem.model.params.Method = 0
                    self.outer_problem.model.update()
                    self.outer_problem.model.optimize()

                    if self.outer_problem.model.status != 2:
                        self.logger.warning(
                            f"Iter {iter_idx}: QP failed with all methods, recovering"
                        )
                        self.outer_problem.recover()
                        # 使用 outer model 的解作为新的 pi_hat
                        pi_hat = np.array(pi_dummy)
                        pi0_hat = pi0_dummy
                        continue

            # 恢复乘子: λ = λ̄ + d*
            d_pi_star = np.array([
                self.outer_problem.d_pi[i].x
                for i in range(config.N_VARS)
            ])
            d_pi0_star = self.outer_problem.d_pi0.x

            pi_hat = self.outer_problem.pi_bar + d_pi_star
            pi0_hat = self.outer_problem.pi0_bar + d_pi0_star

            # 恢复 outer model 为最大化模式
            self.outer_problem.recover()

            # Serious step: 当 LB 有显著提升时，更新稳定中心
            # 判断条件: gap (即当前 dual 值) 显著大于当前 LB
            current_dual = inner_obj - pi_hat @ X_trial - pi0_hat * theta_trial
            if current_dual > LB + 1e-8 * max(1.0, abs(LB)):
                # Serious step: 更新稳定中心
                pi_bar = pi_hat.copy()
                pi0_bar = pi0_hat
                self.outer_problem.set_stability_center(pi_bar, pi0_bar)
                self.logger.info(
                    f"Iter {iter_idx}: Serious step, updated stability center"
                )

            # 日志输出
            self.logger.info(
                f"Iter {iter_idx}: LB={LB:.6f}, UB={UB:.6f}, "
                f"gap={UB - LB:.6e}, pi0_hat={pi0_hat:.6f}, "
                f"d_pi_norm={np.linalg.norm(d_pi_star):.6e}"
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


def load_config_and_solve(
    log=None,
    configs_dir: str = r"D:\tools\workspace_pycharm\SDDiP-RL\bundle_norm_RL\configs",
    # 增量形式参数
    rho: float = 1.0,
    # 对偶边界参数 (均使用默认值时退化为基础算法)
    B_t: float = None,
    norm_bound_type: str = "l1",
    weights: list = None,
) -> SolverResults:
    """
    从保存的 pkl 文件加载 LevelBundleConfig 并求解 (增量形式)

    Args:
        log: 日志器
        configs_dir: config pkl 文件所在目录
        rho: proximal 惩罚参数
        B_t: 对偶边界值 (None 则不添加范数边界约束)
        norm_bound_type: 范数边界约束类型 ("l1" 或 "linf")
        weights: 权重系数列表

    Returns:
        SolverResults
    """
    from bundle_norm_RL.script.config import LevelBundleConfig

    i = 2
    t = 2
    n = 0
    pkl_path = f"{configs_dir}/config_{i}_{t}_{n}.pkl"
    config = LevelBundleConfig.from_pkl(pkl_path)

    log.info(f"Loaded config from {pkl_path}")
    log.info(config.toString())

    solver = LevelBundleSolver(
        log, config, n=n,
        rho=rho, B_t=B_t, norm_bound_type=norm_bound_type,
        weights=weights,
    )
    results = solver.solve()

    log.info(f"Results: {results.toString()}")


if __name__ == "__main__":
    from bundle_norm_RL.script.logger import get_logger
    log = get_logger("../logs/level_bundle_increment.log")
    load_config_and_solve(log)
