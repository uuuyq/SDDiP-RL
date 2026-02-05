import copy
import logging

import numpy as np
import gurobipy as gp

logger = logging.getLogger(__name__)


class Subproblem:
    """
    求解: phi(pi) = min_x L(x, pi)
    该类充当 Oracle，为 Master 提供函数值和子梯度。
    """

    def __init__(self, problem_data):
        self.data = problem_data
        # 实际场景中这里初始化 Gurobi 原问题模型
        # self.model = self.init_model()

    def solve(self, pi: np.ndarray):
        """
        输入: pi (当前对偶变量/乘子)
        返回: phi (函数值), g (子梯度)
        """
        # --- 这里的逻辑对应 DualSolver 中的 get_subgradient_and_value ---
        # 假设原问题是: min f(x) s.t. Ax <= b
        # 拉格朗日函数: L(x, pi) = f(x) + pi * (Ax - b)

        # TODO: 调用 self.model.optimize() 得到 x_star
        # 这里使用 dummy 示例逻辑
        x_star = np.maximum(0, pi)  # 模拟原问题解

        # phi = f(x_star) + pi * (A * x_star - b)
        phi = -0.5 * np.linalg.norm(x_star) ** 2 + pi @ x_star

        # 子梯度 g = Ax_star - b
        g = x_star

        return phi, g


class MasterProblem:
    def __init__(self, n_vars, u_init=1.0, m_l=0.1, m_r=0.5, u_min=1e-5):
        self.n_vars = n_vars
        self.u = u_init
        self.u_min = u_min
        self.m_l = m_l  # Serious step 判定阈值
        self.m_r = m_r  # 权重减小判定阈值
        self.i_u = 0  # 权重连续计数器
        self.var_est = 10 ** 9

        self.model = gp.Model("Master_Bundle")
        self.model.setParam("OutputFlag", 0)
        self.v = self.model.addVar(lb=-gp.GRB.INFINITY, name="v")  # 目标值
        self.pi = self.model.addVars(self.n_vars, lb=-gp.GRB.INFINITY, name="pi")

    def add_cut(self, f_new, g_new, pi_new):
        """添加割平面: v <= f_new + g_new * (pi - pi_new)"""
        cut_expr = f_new + gp.quicksum(g_new[j] * (self.pi[j] - pi_new[j]) for j in range(self.n_vars))
        self.model.addConstr(self.v <= cut_expr)

    def solve_update(self, pi_best):
        """求解主问题获取下一个候选点"""
        prox_term = gp.quicksum((self.pi[j] - pi_best[j]) ** 2 for j in range(self.n_vars))
        self.model.setObjective(self.v - (self.u / 2) * prox_term, gp.GRB.MAXIMIZE)
        self.model.optimize()

        pi_next = np.array([self.pi[j].x for j in range(self.n_vars)])
        v_val = self.v.x
        return pi_next, v_val

    def update_weight(self, pi_new, f_new, pi_best, f_best, v_val, g_new, is_serious):
        """
        对应原代码中的 weight_update 逻辑
        """
        delta = v_val - f_best  # 预测增益
        if delta <= 1e-10: return  # 防止除零

        # 理想权重计算
        u_int = 2 * self.u * (1 - (f_new - f_best) / delta)

        if is_serious:
            # Serious Step 情况
            weight_too_large = (f_new - f_best) >= (self.m_r * delta)
            if weight_too_large and self.i_u > 0:
                self.u = u_int
            elif self.i_u > 3:
                self.u = self.u / 2

            self.u = max(self.u, self.u / 10, self.u_min)
            self.var_est = max(self.var_est, 2 * delta)
            self.i_u = max(self.i_u + 1, 1) if self.u == self.u else 1  # 简化的逻辑计数
        else:
            # Null Step 情况
            p = -self.u * (pi_new - pi_best)
            alpha = delta - np.linalg.norm(p) ** 2 / self.u
            self.var_est = min(self.var_est, np.linalg.norm(p, ord=1) + alpha)

            # 计算线性化误差
            lin_error = f_new + g_new.dot(pi_best - pi_new) - f_best

            if lin_error > max(self.var_est, 10 * delta) and self.i_u < -3:
                self.u = u_int

            self.u = min(self.u, 10 * self.u)
            self.i_u = min(self.i_u - 1, -1)


def run_bundle_method(problem_data, pi_init, max_iter=100, tol=1e-6):
    n_vars = len(pi_init)
    sub = Subproblem(problem_data)
    master = MasterProblem(n_vars)

    pi_best = pi_init
    # 初始评估
    f_best, g_best = sub.solve(pi_best)

    for i in range(max_iter):
        # 1. 求解主问题得到候选点 pi_next 和 预测值 v
        pi_next, v_val = master.solve_update(pi_best)

        # 2. 计算预测增益 delta
        delta = v_val - f_best
        if delta < tol:
            print(f"收敛: 预测增益 {delta} 小于容差")
            break

        # 3. 求解子问题评估候选点
        f_next, g_next = sub.solve(pi_next)

        # 4. 判定是否为 Serious Step
        is_serious = (f_next - f_best) >= master.m_l * delta

        # 5. 更新权重 u
        master.update_weight(pi_next, f_next, pi_best, f_best, v_val, g_next, is_serious)

        # 6. 添加切平面
        master.add_cut(f_next, g_next, pi_next)

        # 7. 如果是 Serious Step，移动中心点
        if is_serious:
            pi_best = pi_next
            f_best = f_next
            print(f"Iter {i}: Serious Step, f = {f_best:.4f}")
        else:
            print(f"Iter {i}: Null Step, delta = {delta:.4f}")

    return pi_best, f_best

class DualSolver:
    def solve(
        self, model: gp.Model, objective_terms, relaxed_terms, normalization=False,
    ) -> tuple[gp.Model, SolverResults]:
        """Solve the dual problem using the bundle method."""
        logger.debug("Bundle method started")
        start_time = time.time()
        time_remaining = self._time_limit

        self.on_solver_call()
        model.setParam("OutputFlag", 0)

        tolerance_reached = False
        time_limit_reached = False

        u = self.u_init
        i_u = 0
        var_est = 10**9

        n_serious_steps = 0

        gradient_len = len(relaxed_terms)
        x_new = np.zeros(gradient_len)
        x_best = np.zeros(gradient_len)

        # Initial subgradient and best lower bound
        subgradient, f_best = self.get_subgradient_and_value(
            model, objective_terms, relaxed_terms, x_best
        )

        f_new = f_best

        # Lowest known gradient magnitude
        lowest_gm = np.linalg.norm(np.array(subgradient))

        # Subproblem with cutting planes
        if normalization:
            subproblem, v, x = self.create_normalized_subproblem(gradient_len)
        else:
            subproblem, v, x = self.create_subproblem(gradient_len)

        for i in range(self.max_iterations):
            # Add new plane to subproblem
            time_remaining = self._get_time_remaining(start_time)

            new_plane = f_new + gp.quicksum(
                subgradient[j] * (x[j] - x_new[j]) for j in range(gradient_len)
            )

            obj = v - u / 2 * gp.quicksum(
                (x[j] - x_best[j]) ** 2 for j in range(gradient_len)
            )
            subproblem.setObjective(obj, gp.GRB.MAXIMIZE)
            subproblem.addConstr(v <= new_plane, name=f"{i+1}")
            subproblem.update()

            subproblem.setParam(
                "TimeLimit",
                max(time_remaining, 10),
                # Ensure that Gurobi has enough time to find at least a
                # feasible point. Otherwise, retrieving the variable
                # values would fail.
            )

            # Solve subproblem
            subproblem.optimize()

            # Candidate dual multipliers
            x_new = [x[j].x for j in range(gradient_len)]

            time_remaining = self._get_time_remaining(start_time)
            # Candidate optimal value
            subgradient, f_new = self.get_subgradient_and_value(
                model, objective_terms, relaxed_terms, x_new, time_remaining
            )

            # Predicted ascent
            delta = self._get_predicted_ascent(f_best, v.x)

            logger.debug("delta = %.8f", delta)

            # Check stopping criterion
            if delta <= self.tolerance:
                tolerance_reached = True
                break
            if time.time() - start_time >= self._time_limit:
                time_limit_reached = True
                break

            # Update lowest known gradient magnitude for logging purposes
            lowest_gm = min(lowest_gm, np.linalg.norm(np.array(subgradient)))

            serious_step = f_new - f_best >= self.m_l * delta

            # Weight update
            u, i_u, var_est = self.weight_update(
                u,
                i_u,
                var_est,
                x_new,
                f_new,
                x_best,
                f_best,
                v.x,
                subgradient,
                serious_step,
            )

            logger.debug(
                "Weight update: u = %.3f, i_u = %s, var_est = %.3f",
                u,
                i_u,
                var_est,
            )

            if serious_step:
                # Serious step
                logger.debug(
                    "Serious step: i = %s, f_new = %.3f, "
                    "f_best = %.3f, f_delta = %.3f, pred_asc = %.3f, "
                    "lowest_gm = %.3f",
                    i + 1,
                    f_new,
                    f_best,
                    f_new - f_best,
                    delta,
                    lowest_gm,
                )
                x_best = copy.copy(x_new)
                f_best = copy.copy(f_new)
                n_serious_steps += 1

        stop_reason = self._get_stop_reason(
            tolerance_reached, time_limit_reached
        )

        self.log_task_end()

        self.log_method_finished(
            stop_reason,
            i + 1,
            lowest_gm,
            f_best,
            n_serious_steps=n_serious_steps,
        )

        self.results.set_values(
            f_best, np.array(x_best), i + 1, self.solver_time
        )

        return (model, self.results)

    def weight_update(
        self,
        u_current,
        i_u,
        var_est,
        x_new,
        f_new,
        x_best,
        f_best,
        f_hat,
        subgradient,
        serious_step,
    ):
        """Update the weight for the bundle method with proximity
        control.
        """
        variation_estimate = var_est

        delta = f_hat - f_best
        u_int = 2 * u_current * (1 - (f_new - f_best) / delta)
        u = u_current

        if serious_step:
            # This is if x_i+1 != x_i
            weight_too_large = (f_new - f_best) >= (self.m_r * delta)
            if weight_too_large and i_u > 0:
                u = u_int
            elif i_u > 3:
                u = u_current / 2
            u_new = max(u, u_current / 10, self.u_min)
            variation_estimate = max(variation_estimate, 2 * delta)
            i_u = max(i_u + 1, 1) if u_new == u_current else 1
            # Exit
        else:
            # This is if x_i+1 = x_i
            p = -u_current * (np.array(x_new) - np.array(x_best))
            alpha = delta - np.linalg.norm(p, ord=2) ** 2 / u_current
            variation_estimate = min(
                variation_estimate, np.linalg.norm(p, ord=1) + alpha
            )
            # x_best x_new Reihenfolge?
            linearization_error = (
                f_new
                + np.array(subgradient).dot(np.array(x_best) - np.array(x_new))
                - f_best
            )
            if (
                linearization_error > max(variation_estimate, 10 * delta)
                and i_u < -3
            ):
                u = u_int
            u_new = min(u, 10 * u_current)
            i_u = min(i_u - 1, -1) if u_new == u_current else -1
            # Exit

        return u_new, i_u, variation_estimate

    def create_subproblem(
        self, n_dual_multipliers: int
    ) -> tuple[gp.Model, gp.Var, gp.tupledict]:
        """Create the bundle method's subproblem."""
        subproblem = gp.Model("Subproblem")
        subproblem.setParam("OutputFlag", 0)
        v = subproblem.addVar(
            vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="v"
        )
        x = subproblem.addVars(
            n_dual_multipliers,
            vtype=gp.GRB.CONTINUOUS,
            lb=-gp.GRB.INFINITY,
            name="x",
        )
        return subproblem, v, x



