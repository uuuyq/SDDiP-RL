"""
ML 模块专用配置文件，包含训练所需的所有参数
"""

import json
import pickle
from pathlib import Path

from sddip.sddip import parameters


class BundleConfig:
    """ML Bundle 配置类，包含所有训练所需参数"""

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
        p_d=None,  # demand realization
        re=None,   # renewable generation realization
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
        self.p_d = p_d  # demand realization features
        self.re = re   # renewable generation realization features

    @property
    def trial_point(self):
        return (self.X_TRIAL, self.Y_TRIAL, self.X_BS_TRIAL, self.SOC_TRIAL)

    def _get_lag_cuts(self):
        """
        解析 Benders cuts 和 Lagrangian cuts，转换为统一的格式
        返回: list of [intercept, g1, g2, ..., gN]
        """
        lag_cuts = []
        
        if self.dual_solver_storage is not None:
            try:
                for s in range(self.PROBLEM_PARAMS.n_stages):
                    lag_result = self.dual_solver_storage.get_stage_result(s)
                    if lag_result:
                        # 获取 Lagrangian cuts
                        cut_intercepts = lag_result.get('dv', [])
                        cut_gradients = lag_result.get('dm', [])
                        
                        for intercept, gradient in zip(cut_intercepts, cut_gradients):
                            # Lagrangian cut 格式: [intercept, g1, g2, ..., gN]
                            cut = [intercept] + (gradient.tolist() if hasattr(gradient, 'tolist') else gradient)
                            lag_cuts.append(cut)
            except Exception as e:
                pass
        
        if self.bc_storage is not None:
            try:
                for s in range(self.PROBLEM_PARAMS.n_stages):
                    benders_result = self.bc_storage.get_stage_result(s)
                    if benders_result:
                        # 获取 Benders cuts
                        bc_intercepts = benders_result.get('bc_intercept', [])
                        bc_gradients = benders_result.get('bc_gradient', [])
                        bc_trial_points = benders_result.get('bc_trial_point', [])
                        
                        for intercept, gradient, trial_point in zip(bc_intercepts, bc_gradients, bc_trial_points):
                            # 转换 Benders cut: intercept = bc_intercept - bc_gradient @ bc_trial_point
                            g = gradient.tolist() if hasattr(gradient, 'tolist') else gradient
                            tp = trial_point.tolist() if hasattr(trial_point, 'tolist') else trial_point
                            converted_intercept = float(intercept) - np.dot(g, tp)
                            cut = [converted_intercept] + g
                            lag_cuts.append(cut)
            except Exception as e:
                pass
        
        return lag_cuts

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
            "p_d": self.p_d,
            "re": self.re,
            "lag_cuts": self._get_lag_cuts(),
        }

    def toString(self):
        """返回JSON字符串表示"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def fromString(cls, json_str):
        """从JSON字符串解析出MLBundleConfig对象"""
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
            p_d=data.get("p_d"),
            re=data.get("re"),
        )

    def to_json(self, file_path):
        """保存为JSON文件"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def to_pkl(self, file_path):
        """保存为pickle文件"""
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pkl(cls, file_path):
        """从pickle文件加载MLBundleConfig对象"""
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def get_realization_features(self):
        """获取realization特征（demand和renewable generation）"""
        return {
            'p_d': self.p_d,
            're': self.re
        }


# 默认配置参数
T = 5
N_VARS = 13

# Trial point
X_TRIAL = [-0.0, 1.0, 1.0]
Y_TRIAL = [0.0, 131.60809087723158, 45.0]
X_BS_TRIAL = [[-0.0, 0.0], [1.0, 1.0], [1.0, 1.0]]
SOC_TRIAL = [0.0]


def get_default_config() -> BundleConfig:
    """获取默认的 MLBundleConfig"""
    return BundleConfig(
        T=T,
        N_VARS=N_VARS,
        X_TRIAL=X_TRIAL,
        Y_TRIAL=Y_TRIAL,
        X_BS_TRIAL=X_BS_TRIAL,
        SOC_TRIAL=SOC_TRIAL,
        PATH=Path(r"D:\tools\workspace_pycharm\SDDiP-RL\data\01_test_cases\case6ww\t06_n06")
    )
