"""
config.py

公共配置文件，包含 trial point 和问题参数
"""

from pathlib import Path

from sddip.sddip import parameters


class BundleConfig:
    """Bundle 配置类，包含所有参数"""

    def __init__(
        self,
        T,
        N_VARS,
        X_TRIAL,
        Y_TRIAL,
        X_BS_TRIAL,
        SOC_TRIAL,
        PROBLEM_PARAMS,
        iteration: int = 0,
        bc_storage=None,
        dual_solver_storage=None,
    ):
        self.T = T
        self.N_VARS = N_VARS
        self.X_TRIAL = X_TRIAL
        self.Y_TRIAL = Y_TRIAL
        self.X_BS_TRIAL = X_BS_TRIAL
        self.SOC_TRIAL = SOC_TRIAL
        self.PROBLEM_PARAMS = PROBLEM_PARAMS
        self.iteration = iteration  # 当前迭代次数 i
        self.bc_storage = bc_storage  # Benders cuts 存储
        self.dual_solver_storage = dual_solver_storage  # Lagrangian cuts 存储

    @property
    def trial_point(self):
        return (self.X_TRIAL, self.Y_TRIAL, self.X_BS_TRIAL, self.SOC_TRIAL)


# 问题参数
T = 5
N_VARS = 13

# Trial point
X_TRIAL = [-0.0, 1.0, 1.0]
Y_TRIAL = [0.0, 131.60809087723158, 45.0]
X_BS_TRIAL = [[-0.0, 0.0], [1.0, 1.0], [1.0, 1.0]]
SOC_TRIAL = [0.0]

# 数据路径
PATH = Path(r"D:\tools\workspace_pycharm\sddip-main-zou\data\01_test_cases\case6ww\t24_n06")

# 初始化 problem_params
PROBLEM_PARAMS = parameters.Parameters(PATH)


def get_default_config() -> BundleConfig:
    """获取默认的 BundleConfig"""
    return BundleConfig(
        T=T,
        N_VARS=N_VARS,
        X_TRIAL=X_TRIAL,
        Y_TRIAL=Y_TRIAL,
        X_BS_TRIAL=X_BS_TRIAL,
        SOC_TRIAL=SOC_TRIAL,
        PROBLEM_PARAMS=PROBLEM_PARAMS,
    )
