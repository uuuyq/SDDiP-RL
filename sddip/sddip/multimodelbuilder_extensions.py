from sddip.sddip.multimodelbuilder import MultiModelBuilder


class MultiModelBuilderExtensions(MultiModelBuilder):
    def relaxed_terms_calculate_without_binary(
            self,
            x_trial_point: list,
            y_trial_point: list,
            x_bs_trial_point: list[list],
            soc_trial_point: list,
            group_id: int = 0
    ):
        """为指定组计算松弛项（不使用二进制变量）"""
        group_vars = self._get_group_variables(group_id)

        self.relaxed_terms = []

        # 计算x的松弛项
        self.relaxed_terms += [
            x_trial_point[j] - group_vars['z_x'][j]
            for j in range(len(x_trial_point))
        ]

        # 计算y的松弛项
        self.relaxed_terms += [
            y_trial_point[j] - group_vars['z_y'][j]
            for j in range(len(y_trial_point))
        ]

        # 计算x_bs的松弛项
        self.relaxed_terms += [
            x_bs_trial_point[g][k] - group_vars['z_x_bs'][g][k]
            for g in range(len(x_bs_trial_point))
            for k in range(len(x_bs_trial_point[g]))
        ]

        # 计算soc的松弛项
        self.relaxed_terms += [
            soc_trial_point[j] - group_vars['z_soc'][j]
            for j in range(len(soc_trial_point))
        ]

    def relaxed_terms_calculate_without_binary_all_groups(
            self,
            x_trial_point: list,
            y_trial_point: list,
            x_bs_trial_point: list[list],
            soc_trial_point: list
    ):
        """为所有组计算松弛项（不使用二进制变量）"""
        all_relaxed_terms = []

        for group_id in range(self.n_groups):
            # 为每个组计算松弛项
            self.relaxed_terms_calculate_without_binary(
                x_trial_point, y_trial_point, x_bs_trial_point, soc_trial_point, group_id
            )
            # 将当前组的松弛项添加到总列表
            all_relaxed_terms.extend(self.relaxed_terms)

        return all_relaxed_terms

    def add_sddip_copy_constraints(
            self,
            x_binary_trial_point: list,
            y_binary_trial_point: list,
            x_bs_binary_trial_point: list[list],
            soc_binary_trial_point: list,
            group_id: int = 0
    ) -> None:
        """为指定组添加SDDIP复制约束"""
        group_vars = self._get_group_variables(group_id)

        self.add_relaxation(
            x_binary_trial_point,
            y_binary_trial_point,
            x_bs_binary_trial_point,
            soc_binary_trial_point,
            group_id
        )
        self.sddip_copy_constrs = []
        for term in self.relaxed_terms:
            self.sddip_copy_constrs.append(
                self.model.addConstr(term == 0, "sddip-copy")
            )

    def add_relaxation(
            self,
            x_binary_trial_point: list,
            y_binary_trial_point: list,
            x_bs_binary_trial_point: list[list],
            soc_binary_trial_point: list,
            group_id: int = 0
    ) -> None:
        """为指定组添加松弛变量"""
        group_vars = self._get_group_variables(group_id)

        self.relaxed_terms = []

        # x的松弛
        self.relaxed_terms += [
            x_binary_trial_point[j] - group_vars['z_x'][j]
            for j in range(len(x_binary_trial_point))
        ]

        # y的松弛
        self.relaxed_terms += [
            y_binary_trial_point[j] - group_vars['z_y'][j]
            for j in range(len(y_binary_trial_point))
        ]

        # x_bs的松弛
        self.relaxed_terms += [
            x_bs_binary_trial_point[g][k] - group_vars['z_x_bs'][g][k]
            for g in range(len(x_bs_binary_trial_point))
            for k in range(len(x_bs_binary_trial_point[g]))
        ]

        # soc的松弛
        self.relaxed_terms += [
            soc_binary_trial_point[j] - group_vars['z_soc'][j]
            for j in range(len(soc_binary_trial_point))
        ]

    def add_sddip_copy_constraints_all_groups(
            self,
            x_binary_trial_point: list,
            y_binary_trial_point: list,
            x_bs_binary_trial_point: list[list],
            soc_binary_trial_point: list
    ) -> None:
        """为所有组添加SDDIP复制约束"""
        all_sddip_copy_constrs = []

        for group_id in range(self.n_groups):
            # 为每个组添加SDDIP复制约束
            self.add_sddip_copy_constraints(
                x_binary_trial_point, y_binary_trial_point, x_bs_binary_trial_point, soc_binary_trial_point, group_id
            )
            # 将当前组的约束添加到总列表
            all_sddip_copy_constrs.extend(self.sddip_copy_constrs)

        self.sddip_copy_constrs = all_sddip_copy_constrs

    def add_relaxation_all_groups(
            self,
            x_binary_trial_point: list,
            y_binary_trial_point: list,
            x_bs_binary_trial_point: list[list],
            soc_binary_trial_point: list
    ) -> list:
        """为所有组添加松弛变量并返回所有松弛项"""
        all_relaxed_terms = []

        for group_id in range(self.n_groups):
            # 为每个组添加松弛
            self.add_relaxation(
                x_binary_trial_point, y_binary_trial_point, x_bs_binary_trial_point, soc_binary_trial_point, group_id
            )
            # 将当前组的松弛项添加到总列表
            all_relaxed_terms.extend(self.relaxed_terms)

        return all_relaxed_terms