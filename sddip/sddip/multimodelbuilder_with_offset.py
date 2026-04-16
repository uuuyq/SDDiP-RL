import gurobipy as gp
import numpy as np
from scipy import linalg


class MultiModelBuilderWithOffset:
    """
    MultiModelBuilder 的扩展版本，支持为所有变量传入偏移值。

    变量变换：
    - x  → x_t + alpha_x
    - y  → y_t + alpha_y
    - x_bs  → x_bs_t + alpha_x_bs
    - soc  → soc_t + alpha_soc
    - z_x  → z_t_x + alpha_z_x
    - z_y  → z_t_y + alpha_z_y
    - z_x_bs  → z_t_x_bs + alpha_z_x_bs
    - z_soc  → z_t_soc + alpha_z_soc

    约束逻辑与原版完全一致，只是变量引用变为偏移量 + alpha变量。
    """

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
        n_groups: int = 1,
        # x 相关偏移值 (每组独立)
        x_t: list = None,           # shape: (n_groups, n_generators)
        y_t: list = None,           # shape: (n_groups, n_generators)
        x_bs_t: list = None,        # shape: (n_groups, n_generators, backsight_periods[g])
        soc_t: list = None,         # shape: (n_groups, n_storages)
        # z 相关偏移值 (每组独立)
        z_t_x: list = None,         # shape: (n_groups, n_generators)
        z_t_y: list = None,         # shape: (n_groups, n_generators)
        z_t_x_bs: list = None,      # shape: (n_groups, n_generators, backsight_periods[g])
        z_t_soc: list = None,       # shape: (n_groups, n_storages)
    ) -> None:
        self.n_buses = n_buses
        self.n_lines = n_lines
        self.n_generators = n_generators
        self.n_storages = n_storages
        self.generators_at_bus = generators_at_bus
        self.storages_at_bus = storages_at_bus
        self.backsight_periods = backsight_periods
        self.n_groups = n_groups
        self.lp_relax = lp_relax

        # 偏移值初始化 (默认全零)
        if x_t is None:
            x_t = [[0.0] * n_generators for _ in range(n_groups)]
        if y_t is None:
            y_t = [[0.0] * n_generators for _ in range(n_groups)]
        if x_bs_t is None:
            x_bs_t = [
                [[0.0] * backsight_periods[g] for g in range(n_generators)]
                for _ in range(n_groups)
            ]
        if soc_t is None:
            soc_t = [[0.0] * n_storages for _ in range(n_groups)]

        if z_t_x is None:
            z_t_x = [[0.0] * n_generators for _ in range(n_groups)]
        if z_t_y is None:
            z_t_y = [[0.0] * n_generators for _ in range(n_groups)]
        if z_t_x_bs is None:
            z_t_x_bs = [
                [[0.0] * backsight_periods[g] for g in range(n_generators)]
                for _ in range(n_groups)
            ]
        if z_t_soc is None:
            z_t_soc = [[0.0] * n_storages for _ in range(n_groups)]

        # 存储偏移值
        self.x_t = x_t
        self.y_t = y_t
        self.x_bs_t = x_bs_t
        self.soc_t = soc_t
        self.z_t_x = z_t_x
        self.z_t_y = z_t_y
        self.z_t_x_bs = z_t_x_bs
        self.z_t_soc = z_t_soc

        # 创建单个Gurobi模型
        self.model = gp.Model("MILP: Multi-Group Unit commitment with Offset")
        self.model.setParam("OutputFlag", 0)
        self.model.setParam("IntFeasTol", 10 ** (-9))
        self.model.setParam("NumericFocus", 3)

        # 使用字典管理多组变量 (alpha变量)
        self.variables = {
            'alpha_x': [],       # 发电机启停调整变量
            'alpha_y': [],       # 发电机出力调整变量
            'alpha_x_bs': [],    # 发电机状态回望调整变量
            'alpha_x_bs_p': [],  # 发电机状态回望的正偏差
            'alpha_x_bs_n': [],  # 发电机状态回望的负偏差
            'alpha_ys_c': [],    # 储能充电功率
            'alpha_ys_dc': [],   # 储能放电功率
            'u_c_dc': [],        # 充放电开关
            'alpha_soc': [],     # 储能荷电状态调整变量
            'alpha_socs_p': [],  # SOC正偏差
            'alpha_socs_n': [],  # SOC负偏差
            'alpha_z_x': [],     # 发电机状态调整变量
            'alpha_z_y': [],     # 发电机出力调整变量
            'alpha_z_x_bs': [],  # 发电机回望状态调整变量
            'alpha_z_soc': [],   # SOC调整变量
            's_up': [],          # 启动决策
            's_down': [],        # 停止决策
            'theta': None,       # 期望值函数近似
            'ys_p': None,        # 正松弛变量
            'ys_n': None,        # 负松弛变量
            'delta': []          # 每个组独立的模型一致性松弛变量
        }

        self.bin_type = gp.GRB.CONTINUOUS if lp_relax else gp.GRB.BINARY

        # 为每个组初始化变量
        for group_id in range(n_groups):
            self._initialize_group_variables(group_id)

        # 为每个组添加 alpha 变量的边界约束
        for group_id in range(n_groups):
            self._add_alpha_bounds_constraints(group_id)

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
            # Commitment decision (alpha_x) - 整数变量，可以是-1
            self.variables['alpha_x'].append(
                self.model.addVar(
                    vtype=gp.GRB.INTEGER, lb=-1, ub=1,
                    name=f"{group_prefix}alpha_x_{g + 1}"
                )
            )

            # Dispatch decision (alpha_y) - 连续变量，无固定下界
            self.variables['alpha_y'].append(
                self.model.addVar(
                    vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY,
                    name=f"{group_prefix}alpha_y_{g + 1}"
                )
            )

            # Generator state backsight variables (alpha_x_bs) - 整数变量，可以是-1
            self.variables['alpha_x_bs'].append(
                [
                    self.model.addVar(
                        vtype=gp.GRB.INTEGER,
                        lb=-1, ub=1,
                        name=f"{group_prefix}alpha_x_bs_{g + 1}_{k + 1}"
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
            self.variables['alpha_x_bs_p'].append(
                [
                    self.model.addVar(
                        vtype=gp.GRB.CONTINUOUS,
                        lb=0, ub=1,
                        name=f"{group_prefix}alpha_x_bs_p_{g + 1}_{k + 1}"
                    )
                    for k in range(self.backsight_periods[g])
                ]
            )
            self.variables['alpha_x_bs_n'].append(
                [
                    self.model.addVar(
                        vtype=gp.GRB.CONTINUOUS,
                        lb=0, ub=1,
                        name=f"{group_prefix}alpha_x_bs_n_{g + 1}_{k + 1}"
                    )
                    for k in range(self.backsight_periods[g])
                ]
            )

            # z 变量 (alpha_z_x, alpha_z_y, alpha_z_x_bs) - 连续变量
            self.variables['alpha_z_x'].append(
                self.model.addVar(
                    vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY,
                    name=f"{group_prefix}alpha_z_x_{g + 1}"
                )
            )
            self.variables['alpha_z_y'].append(
                self.model.addVar(
                    vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY,
                    name=f"{group_prefix}alpha_z_y_{g + 1}"
                )
            )
            self.variables['alpha_z_x_bs'].append(
                [
                    self.model.addVar(
                        vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY,
                        name=f"{group_prefix}alpha_z_x_bs_{g + 1}_{k + 1}",
                    )
                    for k in range(self.backsight_periods[g])
                ]
            )

        for s in range(self.n_storages):
            # Storage charge/discharge
            self.variables['alpha_ys_c'].append(
                self.model.addVar(
                    vtype=gp.GRB.CONTINUOUS, lb=0,
                    name=f"{group_prefix}alpha_y_c_{s+1}"
                )
            )
            self.variables['alpha_ys_dc'].append(
                self.model.addVar(
                    vtype=gp.GRB.CONTINUOUS, lb=0,
                    name=f"{group_prefix}alpha_y_dc_{s+1}"
                )
            )

            # Switch variable
            self.variables['u_c_dc'].append(
                self.model.addVar(
                    vtype=self.bin_type, lb=0, ub=1,
                    name=f"{group_prefix}u_{s+1}"
                )
            )

            # SOC and slack (alpha_soc) - 连续变量，无固定下界
            self.variables['alpha_soc'].append(
                self.model.addVar(
                    vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY,
                    name=f"{group_prefix}alpha_soc_{s+1}"
                )
            )
            self.variables['alpha_socs_p'].append(
                self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=f"{group_prefix}alpha_socs_p")
            )
            self.variables['alpha_socs_n'].append(
                self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=f"{group_prefix}alpha_socs_n")
            )

            # z_soc (alpha_z_soc) - 连续变量，无固定下界
            self.variables['alpha_z_soc'].append(
                self.model.addVar(
                    vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY,
                    name=f"{group_prefix}alpha_z_soc_{s + 1}"
                )
            )

            # Delta variable for this group
            self.variables['delta'].append(
                self.model.addVar(
                    vtype=gp.GRB.CONTINUOUS, lb=0, name=f"{group_prefix}delta"
                )
            )

        # 全局变量（不按组，只有 group_id == 0 时初始化）
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

    def _add_alpha_bounds_constraints(self, group_id: int) -> None:
        """为指定组添加 alpha 变量的边界约束，确保 offset + alpha 在原始变量边界内"""
        self._validate_group_id(group_id)

        start_idx = group_id * self.n_generators
        start_storage = group_id * self.n_storages

        # x: 0 <= x_t + alpha_x <= 1
        for g in range(self.n_generators):
            self.model.addConstr(
                self.x_t[group_id][g] + self.variables['alpha_x'][start_idx + g] >= 0,
                name=f"alpha_x_lb_group_{group_id}_g_{g}"
            )
            self.model.addConstr(
                self.x_t[group_id][g] + self.variables['alpha_x'][start_idx + g] <= 1,
                name=f"alpha_x_ub_group_{group_id}_g_{g}"
            )

        # y: y_t + alpha_y >= 0
        for g in range(self.n_generators):
            self.model.addConstr(
                self.y_t[group_id][g] + self.variables['alpha_y'][start_idx + g] >= 0,
                name=f"alpha_y_lb_group_{group_id}_g_{g}"
            )

        # x_bs: 0 <= x_bs_t + alpha_x_bs <= 1
        for g in range(self.n_generators):
            for k in range(self.backsight_periods[g]):
                self.model.addConstr(
                    self.x_bs_t[group_id][g][k] + self.variables['alpha_x_bs'][start_idx + g][k] >= 0,
                    name=f"alpha_x_bs_lb_group_{group_id}_g_{g}_k_{k}"
                )
                self.model.addConstr(
                    self.x_bs_t[group_id][g][k] + self.variables['alpha_x_bs'][start_idx + g][k] <= 1,
                    name=f"alpha_x_bs_ub_group_{group_id}_g_{g}_k_{k}"
                )

        # soc: soc_t + alpha_soc >= 0
        for s in range(self.n_storages):
            self.model.addConstr(
                self.soc_t[group_id][s] + self.variables['alpha_soc'][start_storage + s] >= 0,
                name=f"alpha_soc_lb_group_{group_id}_s_{s}"
            )

        # z_x: 0 <= z_t_x + alpha_z_x <= 1
        for g in range(self.n_generators):
            self.model.addConstr(
                self.z_t_x[group_id][g] + self.variables['alpha_z_x'][start_idx + g] >= 0,
                name=f"alpha_z_x_lb_group_{group_id}_g_{g}"
            )
            self.model.addConstr(
                self.z_t_x[group_id][g] + self.variables['alpha_z_x'][start_idx + g] <= 1,
                name=f"alpha_z_x_ub_group_{group_id}_g_{g}"
            )

        # z_y: z_t_y + alpha_z_y >= 0
        for g in range(self.n_generators):
            self.model.addConstr(
                self.z_t_y[group_id][g] + self.variables['alpha_z_y'][start_idx + g] >= 0,
                name=f"alpha_z_y_lb_group_{group_id}_g_{g}"
            )

        # z_x_bs: 0 <= z_t_x_bs + alpha_z_x_bs <= 1
        for g in range(self.n_generators):
            for k in range(self.backsight_periods[g]):
                self.model.addConstr(
                    self.z_t_x_bs[group_id][g][k] + self.variables['alpha_z_x_bs'][start_idx + g][k] >= 0,
                    name=f"alpha_z_x_bs_lb_group_{group_id}_g_{g}_k_{k}"
                )
                self.model.addConstr(
                    self.z_t_x_bs[group_id][g][k] + self.variables['alpha_z_x_bs'][start_idx + g][k] <= 1,
                    name=f"alpha_z_x_bs_ub_group_{group_id}_g_{g}_k_{k}"
                )

        # z_soc: z_t_soc + alpha_z_soc >= 0
        for s in range(self.n_storages):
            self.model.addConstr(
                self.z_t_soc[group_id][s] + self.variables['alpha_z_soc'][start_storage + s] >= 0,
                name=f"alpha_z_soc_lb_group_{group_id}_s_{s}"
            )

        self.update_model()

    def _get_group_variables(self, group_id: int) -> dict:
        """获取指定组的所有变量（返回偏移量 + alpha变量的组合）"""
        self._validate_group_id(group_id)

        start_idx = group_id * self.n_generators
        end_idx = (group_id + 1) * self.n_generators
        start_storage = group_id * self.n_storages
        end_storage = (group_id + 1) * self.n_storages

        # 辅助函数：偏移量 + alpha变量
        def make_sum(offset_list, alpha_list, idx):
            return offset_list[idx] + alpha_list[idx]

        def make_sum_2d(offset_list, alpha_list, g, k):
            return offset_list[g][k] + alpha_list[g][k]

        return {
            # x = x_t + alpha_x
            'x': [self.x_t[group_id][g] + self.variables['alpha_x'][start_idx + g] for g in range(self.n_generators)],
            # y = y_t + alpha_y
            'y': [self.y_t[group_id][g] + self.variables['alpha_y'][start_idx + g] for g in range(self.n_generators)],
            # x_bs = x_bs_t + alpha_x_bs
            'x_bs': [
                [self.x_bs_t[group_id][g][k] + self.variables['alpha_x_bs'][start_idx + g][k]
                 for k in range(self.backsight_periods[g])]
                for g in range(self.n_generators)
            ],
            'x_bs_p': self.variables['alpha_x_bs_p'][start_idx:end_idx],
            'x_bs_n': self.variables['alpha_x_bs_n'][start_idx:end_idx],
            's_up': self.variables['s_up'][start_idx:end_idx],
            's_down': self.variables['s_down'][start_idx:end_idx],
            'ys_c': self.variables['alpha_ys_c'][start_storage:end_storage],
            'ys_dc': self.variables['alpha_ys_dc'][start_storage:end_storage],
            'u_c_dc': self.variables['u_c_dc'][start_storage:end_storage],
            # soc = soc_t + alpha_soc
            'soc': [self.soc_t[group_id][s] + self.variables['alpha_soc'][start_storage + s] for s in range(self.n_storages)],
            'socs_p': self.variables['alpha_socs_p'][start_storage:end_storage],
            'socs_n': self.variables['alpha_socs_n'][start_storage:end_storage],
            # z_x = z_t_x + alpha_z_x
            'z_x': [self.z_t_x[group_id][g] + self.variables['alpha_z_x'][start_idx + g] for g in range(self.n_generators)],
            # z_y = z_t_y + alpha_z_y
            'z_y': [self.z_t_y[group_id][g] + self.variables['alpha_z_y'][start_idx + g] for g in range(self.n_generators)],
            # z_x_bs = z_t_x_bs + alpha_z_x_bs
            'z_x_bs': [
                [self.z_t_x_bs[group_id][g][k] + self.variables['alpha_z_x_bs'][start_idx + g][k]
                 for k in range(self.backsight_periods[g])]
                for g in range(self.n_generators)
            ],
            # z_soc = z_t_soc + alpha_z_soc
            'z_soc': [self.z_t_soc[group_id][s] + self.variables['alpha_z_soc'][start_storage + s] for s in range(self.n_storages)],
            # 全局变量
            'ys_p': self.variables['ys_p'],
            'ys_n': self.variables['ys_n'],
            'delta': self.variables['delta'][group_id],
            # 原始 alpha 变量（用于 get_solution）
            'alpha_x': self.variables['alpha_x'][start_idx:end_idx],
            'alpha_y': self.variables['alpha_y'][start_idx:end_idx],
            'alpha_x_bs': self.variables['alpha_x_bs'][start_idx:end_idx],
            'alpha_soc': self.variables['alpha_soc'][start_storage:end_storage],
            'alpha_z_x': self.variables['alpha_z_x'][start_idx:end_idx],
            'alpha_z_y': self.variables['alpha_z_y'][start_idx:end_idx],
            'alpha_z_x_bs': self.variables['alpha_z_x_bs'][start_idx:end_idx],
            'alpha_z_soc': self.variables['alpha_z_soc'][start_storage:end_storage],
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
        """计算松弛项"""
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
        """为指定组添加复制约束

        z_t + alpha_z = x_trial_point
        """
        group_vars = self._get_group_variables(group_id)

        # 复制x约束: z_t_x + alpha_z_x = x_trial_point
        self.constraints['copy']['x'].append(
            self.model.addConstrs(
                (
                    group_vars['z_x'][g] == x_trial_point[g]
                    for g in range(self.n_generators)
                ),
                f"copy-x_group_{group_id}"
            )
        )

        # 复制y约束: z_t_y + alpha_z_y = y_trial_point
        self.constraints['copy']['y'].append(
            self.model.addConstrs(
                (
                    group_vars['z_y'][g] == y_trial_point[g]
                    for g in range(self.n_generators)
                ),
                f"copy-y_group_{group_id}"
            )
        )

        # 复制x_bs约束: z_t_x_bs + alpha_z_x_bs = x_bs_trial_point
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

        # 复制soc约束: z_t_soc + alpha_z_soc = soc_trial_point
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

    def add_cut_lower_bound(self, lower_bound: float) -> None:
        """添加切割下界"""
        self.constraints['cut_lower_bound'] = self.model.addConstr(
            (self.variables['theta'] >= lower_bound), "cut-lb"
        )

    def update_model(self) -> None:
        """更新模型"""
        self.model.update()

    def get_solution(self, group_id: int) -> dict:
        """获取指定组的解"""
        self._validate_group_id(group_id)
        group_vars = self._get_group_variables(group_id)

        return {
            'x': [var.getValue() if hasattr(var, 'getValue') else var for var in group_vars['x']],
            'y': [var.getValue() if hasattr(var, 'getValue') else var for var in group_vars['y']],
            'x_bs': [[var.getValue() if hasattr(var, 'getValue') else var for var in bs_vars] for bs_vars in group_vars['x_bs']],
            'soc': [var.getValue() if hasattr(var, 'getValue') else var for var in group_vars['soc']],
            'z_x': [var.getValue() if hasattr(var, 'getValue') else var for var in group_vars['z_x']],
            'z_y': [var.getValue() if hasattr(var, 'getValue') else var for var in group_vars['z_y']],
            'z_x_bs': [[var.getValue() if hasattr(var, 'getValue') else var for var in bs_vars] for bs_vars in group_vars['z_x_bs']],
            'z_soc': [var.getValue() if hasattr(var, 'getValue') else var for var in group_vars['z_soc']],
            # alpha 变量的值
            'alpha_x': [var.X for var in group_vars['alpha_x']],
            'alpha_y': [var.X for var in group_vars['alpha_y']],
            'alpha_x_bs': [[var.X for var in bs_vars] for bs_vars in group_vars['alpha_x_bs']],
            'alpha_soc': [var.X for var in group_vars['alpha_soc']],
            'alpha_z_x': [var.X for var in group_vars['alpha_z_x']],
            'alpha_z_y': [var.X for var in group_vars['alpha_z_y']],
            'alpha_z_x_bs': [[var.X for var in bs_vars] for bs_vars in group_vars['alpha_z_x_bs']],
            'alpha_z_soc': [var.X for var in group_vars['alpha_z_soc']],
        }
