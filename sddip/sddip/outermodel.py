import gurobipy as gp


class OuterModel:
    def __init__(self, dim_pi, X_trial, theta_trial, benders_pi_list=None):
        self.dim_pi = dim_pi
        self.benders_pi_list = benders_pi_list
        self.subgradient = None
        self.L = None
        self.pi = []
        self.pi0 = None
        self.level_obj = None
        self.lower_bound_const = None
        self.abs_pi = []
        self.beta = []
        self.CL_constrain = []
        self.X_trial = X_trial
        self.theta_trial = theta_trial
        self.model = gp.Model("outer_model")
        self.model.setParam("OutputFlag", 0)

        self.init_model()

    def init_model(self):
        for i in range(self.dim_pi):
            self.pi.append(
                self.model.addVar(
                    vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="pi_%i" % (i + 1)
                )
            )
        self.pi0 = self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=1e-4, name="pi0")

        self.add_l1_norm_constrains()
        # if self.benders_pi_list is not None:
        #     self.add_CL_constrains()

        self.L = self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="L")
        self.model.setObjective(self.L - gp.LinExpr(self.X_trial, self.pi) - self.pi0 * self.theta_trial,
            gp.GRB.MAXIMIZE)
        self.model.update()

    def add_constrains(self, coefficient: list):
        self.model.addConstr(self.L <= gp.LinExpr(coefficient[:-1], self.pi) + self.pi0 * coefficient[-1])
        self.model.update()

    def set_lower_bound(self, lower_bound:float):
        self.lower_bound_const = self.model.addConstr(
            self.L - gp.LinExpr(self.X_trial, self.pi) - self.pi0 * self.theta_trial >= lower_bound,
            name="lower_bound"
        )
        self.model.update()

    def set_level_obj(
        self,
        pi_hat,
        pi0_hat
    ):
        self.level_obj = (gp.quicksum([(self.pi[i] - pi_hat[i]) ** 2 for i in range(len(self.pi))])
                          + (self.pi0 - pi0_hat) ** 2)
        self.model.setObjective(self.level_obj, gp.GRB.MINIMIZE)
        self.model.update()

    def recover(self):
        try:
            self.model.remove(self.lower_bound_const)
        except Exception as e:
            print("recover:", e)
        self.model.setObjective(self.L - gp.LinExpr(self.X_trial, self.pi) - self.pi0 * self.theta_trial, gp.GRB.MAXIMIZE)
        self.model.update()

    # 无穷范数约束
    def add_l8_norm_constrains(self):
        if self.abs_pi:  # 检查是否已经初始化
            print("L1 norm constraints already added. Skipping.")
            return
        for i in range(self.dim_pi):
            self.abs_pi.append(
                self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name="abs_pi_%i" % (i + 1))
            )
        for i in range(self.dim_pi):
            self.model.addConstr(self.abs_pi[i] == gp.abs_(self.pi[i]))
        self.model.addConstr(self.pi0 <= 1, "pi0_cons")
        self.model.update()

    def add_l1_norm_constrains(self):
        for i in range(self.dim_pi):
            self.abs_pi.append(
                self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name="abs_pi_%i" % (i + 1))
            )
        for i in range(self.dim_pi):
            self.model.addConstr(self.abs_pi[i] == gp.abs_(self.pi[i]))
        self.model.addConstr(gp.quicksum(self.abs_pi[i] for i in range(self.dim_pi)) + 20 * self.pi0 <= 1)
        self.model.update()

    def add_CL_constrains(self):
        k = len(self.benders_pi_list)
        for i in range(k):
            self.beta.append(
                self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="beta_%i" % (i + 1))
            )
        # 添加benders pi与lag pi之间的约束，即CL
        for i in range(self.dim_pi):
            self.CL_constrain.append(
                self.model.addConstr(self.pi[i] == gp.quicksum(self.beta[j] * self.benders_pi_list[j][i] for j in range(k)))
            )
        self.model.update()

    def add_constrains_with_offset(self, coefficient: list, offset: float = 0.0):
        """
        添加带常数偏移的切平面约束:
            L <= coefficient[:-1]^T · π + π0 · coefficient[-1] + offset

        正则化场景下 offset = σ · ||z_X_values - X_trial||

        Args:
            coefficient: 次梯度向量 [pi部分..., pi0部分]
            offset: 常数偏移量
        """
        self.model.addConstr(
            self.L <= gp.LinExpr(coefficient[:-1], self.pi) + self.pi0 * coefficient[-1] + offset
        )
        self.model.update()

    def add_l1_norm_bound_constrains(self, B_t: float, weights: list = None):
        """
        L1 范数边界约束 (限制 π 的 L∞ 范数):
            |π_j| ≤ B_t · w_j · π_0, ∀j

        等价于: ||π||_∞ ≤ B_t · ||w||_∞ · π_0

        Args:
            B_t: 对偶边界值
            weights: 权重系数列表，默认全为 1
        """
        if weights is None:
            weights = [1.0] * self.dim_pi

        if not self.abs_pi:
            raise RuntimeError("abs_pi variables not initialized. Call add_l1_norm_constrains or add_l8_norm_constrains first.")

        self.norm_bound_constrs = []
        for j in range(self.dim_pi):
            constr = self.model.addConstr(
                self.abs_pi[j] <= B_t * weights[j] * self.pi0,
                name=f"l1_norm_bound_{j + 1}"
            )
            self.norm_bound_constrs.append(constr)
        self.model.update()

    def add_linf_norm_bound_constrains(self, B_t: float, weights: list = None):
        """
        L∞ 范数边界约束 (限制 π 的加权 L1 范数):
            Σ_j w_j · |π_j| ≤ B_t · π_0

        等价于: ||π||_{w,1} ≤ B_t · π_0

        Args:
            B_t: 对偶边界值
            weights: 权重系数列表，默认全为 1
        """
        if weights is None:
            weights = [1.0] * self.dim_pi

        if not self.abs_pi:
            raise RuntimeError("abs_pi variables not initialized. Call add_l1_norm_constrains or add_l8_norm_constrains first.")

        weighted_abs_pi = gp.quicksum(weights[j] * self.abs_pi[j] for j in range(self.dim_pi))
        self.norm_bound_constrs = [
            self.model.addConstr(
                weighted_abs_pi <= B_t * self.pi0,
                name="linf_norm_bound"
            )
        ]
        self.model.update()

