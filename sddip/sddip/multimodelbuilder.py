import gurobipy as gp
import numpy as np
from scipy import linalg


class MultiModelBuilder:
    def __init__(
        self,
        n_buses: int,
        n_lines: int,
        n_generators: int,
        n_storages: int,
        generators_at_bus: list,
        storages_at_bus: list,
        backsight_periods: list,
        lp_relax: bool = False,
        n_groups: int = 1
    ) -> None:
        """
        初始化MultiModelBuilder

        Args:
            n_groups: 变量组的数量
        """
        self.n_buses = n_buses
        self.n_lines = n_lines
        self.n_generators = n_generators
        self.n_storages = n_storages
        self.generators_at_bus = generators_at_bus
        self.storages_at_bus = storages_at_bus
        self.backsight_periods = backsight_periods
        self.n_groups = n_groups
        self.lp_relax = lp_relax

        # 创建单个Gurobi模型
        self.model = gp.Model("MILP: Multi-Group Unit commitment")
        self.model.setParam("OutputFlag", 0)
        self.model.setParam("IntFeasTol", 10 ** (-9))
        self.model.setParam("NumericFocus", 3)

        # 使用字典管理多组变量
        self.variables = {
            'x': [],       # 发电机启停决策变量
            'y': [],       # 发电机出力变量
            'x_bs': [],    # 发电机状态回望变量
            'x_bs_p': [],  # 发电机状态回望的正负偏差
            'x_bs_n': [],  # 发电机状态回望的负偏差
            'ys_c': [],    # 储能充电功率
            'ys_dc': [],   # 储能放电功率
            'u_c_dc': [],  # 充放电开关
            'soc': [],     # 储能荷电状态
            'socs_p': [],  # SOC正负偏差
            'socs_n': [],  # SOC负偏差
            'z_x': [],     # 发电机状态复制变量
            'z_y': [],     # 发电机出力复制变量
            'z_x_bs': [],  # 发电机回望状态复制变量
            'z_soc': [],   # SOC复制变量
            's_up': [],    # 启动决策
            's_down': [],  # 停止决策
            'theta': None, # 期望值函数近似
            'ys_p': None,  # 正负松弛变量
            'ys_n': None,  # 负松弛变量
            'delta': []    # 每个组独立的模型一致性松弛变量
        }

        self.bin_type = gp.GRB.CONTINUOUS if lp_relax else gp.GRB.BINARY

        # 为每个组初始化变量
        for group_id in range(n_groups):
            self._initialize_group_variables(group_id)

        # 约束管理
        self.constraints = {
            'balance': [],
            'generator': {'min': [], 'max': []},
            'storage': {'charge': [], 'discharge': [], 'soc': []},
            'power_flow': {'line': [], 'line_neg': []},
            'ramp_rate': {'up': [], 'down': []},
            'up_down_time': {'up': [], 'down': [], 'backsight': []},
            'copy': {'x': [], 'y': [], 'x_bs': [], 'soc': []},
            'cut': [],
            'cut_lower_bound': None
        }

        self.objective_terms = None
        self.update_model()

    def _initialize_group_variables(self, group_id: int) -> None:
        """为指定组初始化所有变量"""
        group_prefix = f"group_{group_id}_"
        # 为每个组添加变量
        for g in range(self.n_generators):
            # Commitment decision
            self.variables['x'].append(
                self.model.addVar(
                    vtype=self.bin_type, lb=0, ub=1,
                    name=f"{group_prefix}x_{g + 1}"
                )
            )

            # Dispatch decision
            self.variables['y'].append(
                self.model.addVar(
                    vtype=gp.GRB.CONTINUOUS, lb=0,
                    name=f"{group_prefix}y_{g + 1}"
                )
            )

            # Generator state backsight variables
            self.variables['x_bs'].append(
                [
                    self.model.addVar(
                        vtype=self.bin_type,
                        lb=0, ub=1,
                        name=f"{group_prefix}x_bs_{g + 1}_{k + 1}"
                    )
                    for k in range(self.backsight_periods[g])
                ]
            )

            # Startup/shutdown decisions
            self.variables['s_up'].append(
                self.model.addVar(
                    vtype=self.bin_type, lb=0, ub=1,
                    name=f"{group_prefix}s_up_{g + 1}"
                )
            )
            self.variables['s_down'].append(
                self.model.addVar(
                    vtype=self.bin_type, lb=0, ub=1,
                    name=f"{group_prefix}s_down_{g + 1}"
                )
            )

            # Backsight positive/negative slack
            self.variables['x_bs_p'].append(
                [
                    self.model.addVar(
                        vtype=gp.GRB.CONTINUOUS,
                        lb=0, ub=1,
                        name=f"{group_prefix}x_bs_p_{g + 1}_{k + 1}"
                    )
                    for k in range(self.backsight_periods[g])
                ]
            )
            self.variables['x_bs_n'].append(
                [
                    self.model.addVar(
                        vtype=gp.GRB.CONTINUOUS,
                        lb=0, ub=1,
                        name=f"{group_prefix}x_bs_n_{g + 1}_{k + 1}"
                    )
                    for k in range(self.backsight_periods[g])
                ]
            )

        for s in range(self.n_storages):
            # Storage charge/discharge
            self.variables['ys_c'].append(
                self.model.addVar(
                    vtype=gp.GRB.CONTINUOUS, lb=0,
                    name=f"{group_prefix}y_c_{s+1}"
                )
            )
            self.variables['ys_dc'].append(
                self.model.addVar(
                    vtype=gp.GRB.CONTINUOUS, lb=0,
                    name=f"{group_prefix}y_dc_{s+1}"
                )
            )

            # Switch variable
            self.variables['u_c_dc'].append(
                self.model.addVar(
                    vtype=self.bin_type, lb=0, ub=1,
                    name=f"{group_prefix}u_{s+1}"
                )
            )

            # SOC and slack
            self.variables['soc'].append(
                self.model.addVar(
                    vtype=gp.GRB.CONTINUOUS, lb=0,
                    name=f"{group_prefix}soc_{s+1}"
                )
            )
            self.variables['socs_p'].append(
                self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=f"{group_prefix}socs_p")
            )
            self.variables['socs_n'].append(
                self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=f"{group_prefix}socs_n")
            )

            # Delta variable for this group
            self.variables['delta'].append(
                self.model.addVar(
                    vtype=gp.GRB.CONTINUOUS, lb=0, name=f"{group_prefix}delta"
                )
            )

        # 全局变量（不按组）
        if group_id == 0:
            self.variables['theta'] = self.model.addVar(
                vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="theta"
            )
            self.variables['ys_p'] = self.model.addVar(
                vtype=gp.GRB.CONTINUOUS, lb=0, name="ys_p"
            )
            self.variables['ys_n'] = self.model.addVar(
                vtype=gp.GRB.CONTINUOUS, lb=0, name="ys_n"
            )

            # Copy variables
            for g in range(self.n_generators):
                self.variables['z_x'].append(
                    self.model.addVar(
                        vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, name=f"z_x_{g + 1}"
                    )
                )
                self.variables['z_y'].append(
                    self.model.addVar(
                        vtype=gp.GRB.CONTINUOUS, name=f"z_y_{g + 1}"
                    )
                )
                self.variables['z_x_bs'].append(
                    [
                        self.model.addVar(
                            vtype=gp.GRB.CONTINUOUS, lb=0, ub=1,
                            name=f"z_x_bs_{g + 1}_{k + 1}",
                        )
                        for k in range(self.backsight_periods[g])
                    ]
                )
            for s in range(self.n_storages):
                self.variables['z_soc'].append(
                    self.model.addVar(
                        vtype=gp.GRB.CONTINUOUS, name=f"z_soc_{s + 1}"
                    )
                )

    def _get_group_variables(self, group_id: int) -> dict:
        """获取指定组的所有变量"""
        self._validate_group_id(group_id)

        start_idx = group_id * self.n_generators
        end_idx = (group_id + 1) * self.n_generators

        return {
            'x': self.variables['x'][start_idx:end_idx],
            'y': self.variables['y'][start_idx:end_idx],
            'x_bs': self.variables['x_bs'][start_idx:end_idx],
            'x_bs_p': self.variables['x_bs_p'][start_idx:end_idx],
            'x_bs_n': self.variables['x_bs_n'][start_idx:end_idx],
            's_up': self.variables['s_up'][start_idx:end_idx],
            's_down': self.variables['s_down'][start_idx:end_idx],
            'ys_c': self.variables['ys_c'][group_id * self.n_storages:(group_id + 1) * self.n_storages],
            'ys_dc': self.variables['ys_dc'][group_id * self.n_storages:(group_id + 1) * self.n_storages],
            'u_c_dc': self.variables['u_c_dc'][group_id * self.n_storages:(group_id + 1) * self.n_storages],
            'soc': self.variables['soc'][group_id * self.n_storages:(group_id + 1) * self.n_storages],
            'socs_p': self.variables['socs_p'][group_id * self.n_storages:(group_id + 1) * self.n_storages],
            'socs_n': self.variables['socs_n'][group_id * self.n_storages:(group_id + 1) * self.n_storages],
            'z_x': self.variables['z_x'],
            'z_y': self.variables['z_y'],
            'z_x_bs': self.variables['z_x_bs'],
            'z_soc': self.variables['z_soc'],
            'ys_p': self.variables['ys_p'],
            'ys_n': self.variables['ys_n'],
            'delta': self.variables['delta'][group_id]
        }

    def _validate_group_id(self, group_id: int) -> None:
        """验证组ID是否有效"""
        if group_id < 0 or group_id >= self.n_groups:
            raise ValueError(f"Invalid group ID: {group_id}. Must be between 0 and {self.n_groups - 1}")

    # ========== 约束添加方法 ==========

    def add_balance_constraints(
        self,
        total_demand: float,
        total_renewable_generation: float,
        discharge_eff: list,
        group_id: int = 0
    ) -> None:
        """为指定组添加平衡约束"""
        group_vars = self._get_group_variables(group_id)

        self.constraints['balance'].append(
            self.model.addConstr(
                gp.quicksum(group_vars['y'])
                + gp.quicksum(
                    discharge_eff[s] * group_vars['ys_dc'][s] - group_vars['ys_c'][s]
                    for s in range(self.n_storages)
                )
                + group_vars['ys_p']
                - group_vars['ys_n']
                == total_demand - total_renewable_generation,
                f"balance_group_{group_id}"
            )
        )
        self.update_model()

    def add_generator_constraints(
        self, min_generation: list, max_generation: list, group_id: int = 0
    ) -> None:
        """为指定组添加发电机约束"""
        group_vars = self._get_group_variables(group_id)

        self.constraints['generator']['min'].append(
            self.model.addConstrs(
                (
                    group_vars['y'][g] >= min_generation[g] * group_vars['x'][g] - group_vars['delta']
                    for g in range(self.n_generators)
                ),
                f"min-generation_group_{group_id}"
            )
        )

        self.constraints['generator']['max'].append(
            self.model.addConstrs(
                (
                    group_vars['y'][g] <= max_generation[g] * group_vars['x'][g] + group_vars['delta']
                    for g in range(self.n_generators)
                ),
                f"max-generation_group_{group_id}"
            )
        )
        self.update_model()

    def add_storage_constraints(
        self, max_charge_rate: list, max_discharge_rate: list, max_soc: list, group_id: int = 0
    ) -> None:
        """为指定组添加储能约束"""
        group_vars = self._get_group_variables(group_id)

        self.constraints['storage']['charge'].append(
            self.model.addConstrs(
                (
                    group_vars['ys_c'][s] <= max_charge_rate[s] * group_vars['u_c_dc'][s]
                    for s in range(self.n_storages)
                ),
                f"max-charge-rate_group_{group_id}"
            )
        )

        self.constraints['storage']['discharge'].append(
            self.model.addConstrs(
                (
                    group_vars['ys_dc'][s] <= max_discharge_rate[s] * (1 - group_vars['u_c_dc'][s])
                    for s in range(self.n_storages)
                ),
                f"max-discharge-rate_group_{group_id}"
            )
        )

        self.constraints['storage']['soc'].append(
            self.model.addConstrs(
                (
                    group_vars['soc'][s] <= max_soc[s] + group_vars['delta']
                    for s in range(self.n_storages)
                ),
                f"max-soc_group_{group_id}"
            )
        )
        self.update_model()

    def add_soc_transfer(self, charge_eff: list, group_id: int = 0) -> None:
        """为指定组添加SOC传递约束"""
        group_vars = self._get_group_variables(group_id)

        self.constraints['storage']['soc'].append(
            self.model.addConstrs(
                (
                    group_vars['soc'][s]
                    == group_vars['z_soc'][s]
                    + charge_eff[s] * group_vars['ys_c'][s]
                    - group_vars['ys_dc'][s]
                    + group_vars['socs_p'][s]
                    - group_vars['socs_n'][s]
                    for s in range(self.n_storages)
                ),
                f"soc_group_{group_id}"
            )
        )
        self.update_model()

    def add_final_soc_constraints(self, final_soc: list, group_id: int = 0) -> None:
        """为指定组添加最终SOC约束"""
        group_vars = self._get_group_variables(group_id)

        self.constraints['storage']['soc'].append(
            self.model.addConstrs(
                (
                    group_vars['soc'][s] >= final_soc[s] - group_vars['delta']
                    for s in range(self.n_storages)
                ),
                f"final soc_group_{group_id}"
            )
        )
        self.update_model()

    def add_power_flow_constraints(
        self,
        ptdf,
        max_line_capacities: list,
        demand: list,
        renewable_generation: list,
        discharge_eff: list,
        group_id: int = 0
    ) -> None:
        """为指定组添加电力流约束"""
        group_vars = self._get_group_variables(group_id)

        line_flows = [
            gp.quicksum(
                ptdf[l, b]
                * (
                    gp.quicksum(group_vars['y'][g] for g in self.generators_at_bus[b])
                    + gp.quicksum(
                        discharge_eff[s] * group_vars['ys_dc'][s] - group_vars['ys_c'][s]
                        for s in self.storages_at_bus[b]
                    )
                    - demand[b]
                    + renewable_generation[b]
                )
                for b in range(self.n_buses)
            )
            for l in range(self.n_lines)
        ]

        self.constraints['power_flow']['line'].append(
            self.model.addConstrs(
                (
                    line_flows[l] <= max_line_capacities[l] + group_vars['delta']
                    for l in range(self.n_lines)
                ),
                f"power-flow(1)_group_{group_id}"
            )
        )

        self.constraints['power_flow']['line_neg'].append(
            self.model.addConstrs(
                (
                    -line_flows[l] <= max_line_capacities[l] + group_vars['delta']
                    for l in range(self.n_lines)
                ),
                f"power-flow(2)_group_{group_id}"
            )
        )
        self.update_model()

    def add_startup_shutdown_constraints(self, group_id: int = 0) -> None:
        """为指定组添加启停约束"""
        group_vars = self._get_group_variables(group_id)

        self.constraints['up_down_time']['up'].append(
            self.model.addConstrs(
                (
                    group_vars['x'][g] - group_vars['z_x'][g] <= group_vars['s_up'][g] + group_vars['delta']
                    for g in range(self.n_generators)
                ),
                f"start-up_group_{group_id}"
            )
        )

        self.constraints['up_down_time']['down'].append(
            self.model.addConstrs(
                (
                    group_vars['z_x'][g] - group_vars['x'][g] <= group_vars['s_down'][g] + group_vars['delta']
                    for g in range(self.n_generators)
                ),
                f"shut-down_group_{group_id}"
            )
        )
        self.update_model()

    def add_ramp_rate_constraints(
        self,
        max_rate_up: list,
        max_rate_down: list,
        startup_rate: list,
        shutdown_rate: list,
        group_id: int = 0
    ) -> None:
        """为指定组添加爬坡率约束"""
        group_vars = self._get_group_variables(group_id)

        self.constraints['ramp_rate']['up'].append(
            self.model.addConstrs(
                (
                    group_vars['y'][g] - group_vars['z_y'][g]
                    <= max_rate_up[g] * group_vars['z_x'][g]
                    + startup_rate[g] * group_vars['s_up'][g]
                    + group_vars['delta']
                    for g in range(self.n_generators)
                ),
                f"rate-up_group_{group_id}"
            )
        )

        self.constraints['ramp_rate']['down'].append(
            self.model.addConstrs(
                (
                    group_vars['z_y'][g] - group_vars['y'][g]
                    <= max_rate_down[g] * group_vars['x'][g]
                    + shutdown_rate[g] * group_vars['s_down'][g]
                    + group_vars['delta']
                    for g in range(self.n_generators)
                ),
                f"rate-down_group_{group_id}"
            )
        )
        self.update_model()

    def add_up_down_time_constraints(
        self, min_up_times: list, min_down_times: list, group_id: int = 0
    ) -> None:
        """为指定组添加最小启停时间约束"""
        group_vars = self._get_group_variables(group_id)

        self.constraints['up_down_time']['up'].append(
            self.model.addConstrs(
                (
                    gp.quicksum(group_vars['z_x_bs'][g])
                    >= min_up_times[g] * group_vars['s_down'][g] - group_vars['delta']
                    for g in range(self.n_generators)
                ),
                f"up-time_group_{group_id}"
            )
        )

        self.constraints['up_down_time']['down'].append(
            self.model.addConstrs(
                (
                    len(group_vars['z_x_bs'][g]) - gp.quicksum(group_vars['z_x_bs'][g])
                    >= min_down_times[g] * group_vars['s_up'][g] - group_vars['delta']
                    for g in range(self.n_generators)
                ),
                f"down-time_group_{group_id}"
            )
        )

        self.constraints['up_down_time']['backsight'].append(
            self.model.addConstrs(
                (
                    group_vars['z_x_bs'][g][k]
                    == group_vars['x_bs'][g][k] + group_vars['x_bs_p'][g][k] - group_vars['x_bs_n'][g][k]
                    for g in range(self.n_generators)
                    for k in range(self.backsight_periods[g])
                ),
                f"backsight_group_{group_id}"
            )
        )
        self.update_model()


    def relaxed_terms_calculate_without_binary(
            self,
            x_trial_point: list,
            y_trial_point: list,
            x_bs_trial_point: list[list],
            soc_trial_point: list,
            group_id: int = 0
    ):
        group_vars = self._get_group_variables(group_id)
        relaxed_terms = []
        relaxed_terms += [
            x_trial_point[j] - group_vars["z_x"][j]
            for j in range(len(x_trial_point))
        ]

        relaxed_terms += [
            y_trial_point[j] - group_vars["z_y"][j]
            for j in range(len(y_trial_point))
        ]

        relaxed_terms += [
            x_bs_trial_point[g][k] - group_vars["z_x_bs"][g][k]
            for g in range(len(x_bs_trial_point))
            for k in range(len(x_bs_trial_point[g]))
        ]

        relaxed_terms += [
            soc_trial_point[j] - group_vars["z_soc"][j]
            for j in range(len(soc_trial_point))
        ]

        return relaxed_terms

    def add_copy_constraints(
        self,
        x_trial_point: list,
        y_trial_point: list,
        x_bs_trial_point: list[list],
        soc_trial_point: list,
        group_id: int = 0
    ) -> None:
        """为指定组添加复制约束"""
        group_vars = self._get_group_variables(group_id)

        # 复制x约束
        self.constraints['copy']['x'].append(
            self.model.addConstrs(
                (
                    group_vars['z_x'][g] == x_trial_point[g]
                    for g in range(self.n_generators)
                ),
                f"copy-x_group_{group_id}"
            )
        )

        # 复制y约束
        self.constraints['copy']['y'].append(
            self.model.addConstrs(
                (
                    group_vars['z_y'][g] == y_trial_point[g]
                    for g in range(self.n_generators)
                ),
                f"copy-y_group_{group_id}"
            )
        )

        # 复制x_bs约束
        self.constraints['copy']['x_bs'].append(
            self.model.addConstrs(
                (
                    group_vars['z_x_bs'][g][k] == x_bs_trial_point[g][k]
                    for g in range(self.n_generators)
                    for k in range(self.backsight_periods[g])
                ),
                f"copy-x-bs_group_{group_id}"
            )
        )

        # 复制soc约束
        self.constraints['copy']['soc'].append(
            self.model.addConstrs(
                (
                    group_vars['z_soc'][s] == soc_trial_point[s]
                    for s in range(self.n_storages)
                ),
                f"copy-soc_group_{group_id}"
            )
        )
        self.update_model()

    # def add_benders_cuts(
    #     self, cut_intercepts: list, cut_gradients: list, trial_points: list, group_id: int = 0
    # ) -> None:
    #     """为指定组添加Benders切割"""
    #     group_vars = self._get_group_variables(group_id)
    #
    #     state_variables = (
    #         group_vars['x']
    #         + group_vars['y']
    #         + [var for gen_bs in group_vars['x_bs'] for var in gen_bs]
    #         + group_vars['soc']
    #     )
    #
    #     n_state_variables = len(state_variables)
    #
    #     for intercept, gradient, trial_point in zip(
    #         cut_intercepts, cut_gradients, trial_points, strict=False
    #     ):
    #         if n_state_variables != len(trial_point):
    #             msg = "Number of state variables must be equal to the number of trial points."
    #             raise ValueError(msg)
    #
    #         self.model.addConstr(
    #             (
    #                 self.variables['theta']
    #                 >= intercept
    #                 + gp.quicksum(
    #                     gradient[i] * (state_variables[i] - trial_point[i])
    #                     for i in range(n_state_variables)
    #                 )
    #             ),
    #             f"cut_group_{group_id}"
    #         )
    #
    #     self.update_model()

    def add_cut_lower_bound(self, lower_bound: float) -> None:
        """添加切割下界"""
        self.constraints['cut_lower_bound'] = self.model.addConstr(
            (self.variables['theta'] >= lower_bound), "cut-lb"
        )

    # # ========== 目标函数 ==========
    #
    # def add_objective(self, coefficients: list, group_id: int = 0) -> None:
    #     """为指定组添加目标函数"""
    #     group_vars = self._get_group_variables(group_id)
    #
    #     x_bs_p = [x for g in range(self.n_generators) for x in group_vars['x_bs_p'][g]]
    #     x_bs_n = [x for g in range(self.n_generators) for x in group_vars['x_bs_n'][g]]
    #
    #     penalty = coefficients[-1]
    #     coefficients = (
    #         coefficients
    #         + [penalty] * (2 * self.n_storages + 2 * len(x_bs_p) + 1)
    #         + [1]
    #     )
    #
    #     variables = (
    #         group_vars['y']
    #         + group_vars['s_up']
    #         + group_vars['s_down']
    #         + [group_vars['ys_p'], group_vars['ys_n']]
    #         + group_vars['socs_p']
    #         + group_vars['socs_n']
    #         + x_bs_p
    #         + x_bs_n
    #         + [group_vars['delta']]
    #         + [self.variables['theta']]
    #     )
    #
    #     self.objective_terms = gp.LinExpr(coefficients, variables)
    #     self.model.setObjective(self.objective_terms)
    #     self.update_model()


    def update_model(self) -> None:
        """更新模型"""
        self.model.update()


    # def get_group_variable_names(self, group_id: int) -> dict:
    #     """获取指定组的变量名"""
    #     self._validate_group_id(group_id)
    #     group_vars = self._get_group_variables(group_id)
    #     return {
    #         'x': [var.VarName for var in group_vars['x']],
    #         'y': [var.VarName for var in group_vars['y']],
    #         'x_bs': [[var.VarName for var in bs_vars] for bs_vars in group_vars['x_bs']],
    #         'soc': [var.VarName for var in group_vars['soc']]
    #     }

    # def solve(self) -> None:
    #     """求解模型"""
    #     self.model.optimize()

    def get_solution(self, group_id: int) -> dict:
        """获取指定组的解"""
        self._validate_group_id(group_id)
        group_vars = self._get_group_variables(group_id)

        return {
            'x': [var.X for var in group_vars['x']],
            'y': [var.X for var in group_vars['y']],
            'x_bs': [[var.X for var in bs_vars] for bs_vars in group_vars['x_bs']],
            'soc': [var.X for var in group_vars['soc']]
        }