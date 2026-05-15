"""
公共配置文件，包含 trial point 和问题参数
"""

import json
import pickle
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
        PATH,
        iteration: int = 0,
        bc_storage=None,
        dual_solver_storage=None,
        n: int = 0,  # realization 索引
    ):
        self.T = T
        self.N_VARS = N_VARS
        self.X_TRIAL = X_TRIAL
        self.Y_TRIAL = Y_TRIAL
        self.X_BS_TRIAL = X_BS_TRIAL
        self.SOC_TRIAL = SOC_TRIAL
        self.PATH = PATH if isinstance(PATH, str) else str(PATH)
        self.PROBLEM_PARAMS = parameters.Parameters(PATH)
        self.iteration = iteration  # 当前迭代次数 i
        self.bc_storage = bc_storage  # Benders cuts 存储
        self.dual_solver_storage = dual_solver_storage  # Lagrangian cuts 存储
        self.n = n  # realization 索引

    @property
    def trial_point(self):
        return (self.X_TRIAL, self.Y_TRIAL, self.X_BS_TRIAL, self.SOC_TRIAL)

    def to_dict(self):
        """将对象转换为字典，用于JSON序列化"""
        return {
            "T": self.T,
            "N_VARS": self.N_VARS,
            "X_TRIAL": self.X_TRIAL,
            "Y_TRIAL": self.Y_TRIAL,
            "X_BS_TRIAL": self.X_BS_TRIAL,
            "SOC_TRIAL": self.SOC_TRIAL,
            "PATH": self.PATH,
            "iteration": self.iteration,
            "n": self.n,
            "bc_storage_type": type(self.bc_storage).__name__ if self.bc_storage else None,
            "dual_solver_storage_type": type(self.dual_solver_storage).__name__ if self.dual_solver_storage else None,
        }

    def toString(self):
        """返回JSON字符串表示"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def fromString(cls, json_str):
        """从JSON字符串解析出BundleConfig对象"""
        data = json.loads(json_str)
        return cls(
            T=data["T"],
            N_VARS=data["N_VARS"],
            X_TRIAL=data["X_TRIAL"],
            Y_TRIAL=data["Y_TRIAL"],
            X_BS_TRIAL=data["X_BS_TRIAL"],
            SOC_TRIAL=data["SOC_TRIAL"],
            PATH=data["PATH"],
            iteration=data.get("iteration", 0),
            n=data.get("n", 0),
        )

    def to_pkl(self, file_path):
        """保存为pickle文件"""
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pkl(cls, file_path):
        """从pickle文件加载BundleConfig对象"""
        with open(file_path, "rb") as f:
            return pickle.load(f)


# 问题参数
T = 5
N_VARS = 13

# Trial point
X_TRIAL = [-0.0, 1.0, 1.0]
Y_TRIAL = [0.0, 131.60809087723158, 45.0]
X_BS_TRIAL = [[-0.0, 0.0], [1.0, 1.0], [1.0, 1.0]]
SOC_TRIAL = [0.0]

# 数据路径
PATH = Path(r"D:\tools\workspace_pycharm\sddip-main-zou\data\01_test_cases\case6ww\t06_n06")



def get_default_config() -> BundleConfig:
    """获取默认的 BundleConfig"""
    return BundleConfig(
        T=T,
        N_VARS=N_VARS,
        X_TRIAL=X_TRIAL,
        Y_TRIAL=Y_TRIAL,
        X_BS_TRIAL=X_BS_TRIAL,
        SOC_TRIAL=SOC_TRIAL,
        PATH=PATH
    )