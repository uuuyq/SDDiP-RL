"""
公共配置文件，包含 trial point 和问题参数
"""

import json
import pickle
from pathlib import Path

from sddip.sddip import parameters



class LevelBundleConfig:
    """Level Bundle 配置类，包含所有参数"""

    def __init__(
        self,
        T,
        N_VARS,
        X_TRIAL,
        Y_TRIAL,
        X_BS_TRIAL,
        SOC_TRIAL,
        THETA_TRIAL,
        PATH,
        iteration: int = 0,
        bc_storage=None,
        dual_solver_storage=None,
        n: int = 0,  # realization 索引
        # Level Bundle 专用参数
        level_factor: float = 0.3,
        iteration_limit: int = 200,
        gap_tol: float = 5e-3,
        pi0_tol: float = 1e-8,
        time_limit: int = 3600,  # 秒
    ):
        self.T = T
        self.N_VARS = N_VARS
        self.X_TRIAL = X_TRIAL
        self.Y_TRIAL = Y_TRIAL
        self.X_BS_TRIAL = X_BS_TRIAL
        self.SOC_TRIAL = SOC_TRIAL
        self.THETA_TRIAL = THETA_TRIAL
        self.PATH = PATH if isinstance(PATH, str) else str(PATH)
        self.PROBLEM_PARAMS = parameters.Parameters(Path(PATH))
        self.iteration = iteration  # 当前迭代次数 i
        self.bc_storage = bc_storage  # Benders cuts 存储
        self.dual_solver_storage = dual_solver_storage  # Lagrangian cuts 存储
        self.n = n  # realization 索引
        # Level Bundle 专用参数
        self.level_factor = level_factor
        self.iteration_limit = iteration_limit
        self.gap_tol = gap_tol
        self.pi0_tol = pi0_tol
        self.time_limit = time_limit

    @property
    def trial_point(self):
        return (self.X_TRIAL, self.Y_TRIAL, self.X_BS_TRIAL, self.SOC_TRIAL)

    @property
    def X_trial(self):
        """拼接后的 trial point: x + y + x_bs_flat + soc"""
        return (
            self.X_TRIAL
            + self.Y_TRIAL
            + [val for bs in self.X_BS_TRIAL for val in bs]
            + self.SOC_TRIAL
        )

    def to_dict(self):
        """将对象转换为字典，用于JSON序列化"""
        return {
            "T": self.T,
            "N_VARS": self.N_VARS,
            "X_TRIAL": self.X_TRIAL,
            "Y_TRIAL": self.Y_TRIAL,
            "X_BS_TRIAL": self.X_BS_TRIAL,
            "SOC_TRIAL": self.SOC_TRIAL,
            "THETA_TRIAL": self.THETA_TRIAL,
            "PATH": self.PATH,
            "iteration": self.iteration,
            "n": self.n,
            "level_factor": self.level_factor,
            "iteration_limit": self.iteration_limit,
            "gap_tol": self.gap_tol,
            "pi0_tol": self.pi0_tol,
            "time_limit": self.time_limit,
            "bc_storage_type": type(self.bc_storage).__name__ if self.bc_storage else None,
            "dual_solver_storage_type": type(self.dual_solver_storage).__name__ if self.dual_solver_storage else None,
        }

    def toString(self):
        """返回JSON字符串表示"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def fromString(cls, json_str):
        """从JSON字符串解析出LevelBundleConfig对象"""
        data = json.loads(json_str)
        return cls(
            T=data["T"],
            N_VARS=data["N_VARS"],
            X_TRIAL=data["X_TRIAL"],
            Y_TRIAL=data["Y_TRIAL"],
            X_BS_TRIAL=data["X_BS_TRIAL"],
            SOC_TRIAL=data["SOC_TRIAL"],
            THETA_TRIAL=data["THETA_TRIAL"],
            PATH=data["PATH"],
            iteration=data.get("iteration", 0),
            n=data.get("n", 0),
            level_factor=data.get("level_factor", 0.3),
            iteration_limit=data.get("iteration_limit", 200),
            gap_tol=data.get("gap_tol", 5e-3),
            pi0_tol=data.get("pi0_tol", 1e-4),
            time_limit=data.get("time_limit", 3600),
        )

    def to_pkl(self, file_path):
        """保存为pickle文件"""
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pkl(cls, file_path):
        """从pickle文件加载LevelBundleConfig对象"""
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
THETA_TRIAL = 0.0


def get_default_level_bundle_config() -> LevelBundleConfig:
    """获取默认的 LevelBundleConfig"""
    return LevelBundleConfig(
        T=T,
        N_VARS=N_VARS,
        X_TRIAL=X_TRIAL,
        Y_TRIAL=Y_TRIAL,
        X_BS_TRIAL=X_BS_TRIAL,
        SOC_TRIAL=SOC_TRIAL,
        THETA_TRIAL=THETA_TRIAL,
        PATH=Path(r"D:\tools\workspace_pycharm\SDDiP-RL\data\01_test_cases\case6ww\t06_n06")
    )
