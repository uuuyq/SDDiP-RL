
"""
基于 ML 的 Lagrangian 子问题求解器
集成 Neural Warm Start 模型
"""
import logging
import numpy as np
import torch

from bundle_ml.models import NeuralWarmStartModel
from bundle_ml.ml_config import MLConfig
from bundle_ml.lag_problem import SubProblem, MasterProblem

logger = logging.getLogger(__name__)

@DeprecationWarning
class MLSubProblem(SubProblem):
    """
    基于 ML 的子问题求解器

    支持两种模式:
    - "replace": 完全用 ML 模型替代 Gurobi 求解
    - "warm_start": 用 ML 模型提供初始值，再用 Gurobi 求解
    """

    def __init__(
        self,
        logger,
        config,
        n,
        i_override=None,
        model_path: str = None,
        mode: str = "replace",
        device: str = "cpu",
    ):
        """
        Args:
            logger: 日志器
            config: BundleConfig 对象
            n: realization 索引
            i_override: 可选，覆盖 config 中的 iteration
            model_path: 预训练模型路径
            mode: "replace" 或 "warm_start"
            device: 设备 ("cpu" 或 "cuda")
        """
        # 先初始化父类
        super().__init__(logger, config, n, i_override)

        self.mode = mode
        self.device = device

        # 加载 ML 模型
        if model_path is not None:
            self.model = NeuralWarmStartModel.load(model_path)
        else:
            self.model = NeuralWarmStartModel()

        self.model.to(device)
        self.model.eval()

        # 配置
        self.ml_config = MLConfig()
        self.max_cuts = self.ml_config.data["max_cuts"]

    def solve(
        self,
        pi: np.ndarray,
        time_limit: float = None,
        master: MasterProblem = None,
    ):
        """
        求解子问题

        Args:
            pi: 当前对偶变量
            time_limit: 时间限制
            master: MasterProblem 对象，用于获取 cuts

        Returns:
            subgradient: 子梯度
            opt_value: 最优值
        """
        if self.mode == "replace":
            # 完全用 ML 模型替代
            return self._solve_with_ml(pi, master)
        elif self.mode == "warm_start":
            # 先用 ML 提供初始值，再用 Gurobi 求解
            return self._solve_with_warm_start(pi, time_limit, master)
        else:
            # 回退到原始 Gurobi 求解
            return super().solve(pi, time_limit)

    def _solve_with_ml(
        self,
        pi: np.ndarray,
        master: MasterProblem = None,
    ):
        """
        完全用 ML 模型求解

        Args:
            pi: 当前对偶变量
            master: MasterProblem 对象

        Returns:
            subgradient: 子梯度
            opt_value: 最优值
        """
        with torch.no_grad():
            # 1. 准备输入数据
            batch_size = 1

            # Lambda
            lambda_tensor = torch.tensor(pi, dtype=torch.float32).unsqueeze(0).to(self.device)

            # x_prev (trial_point)
            x_prev = self._flatten_trial_point(self.config.trial_point)
            x_prev_tensor = torch.tensor(x_prev, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Realization (demand + renewable)
            realization = self._get_realization_features()
            realization_tensor = torch.tensor(realization, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Stage
            stage_tensor = torch.tensor([self.config.T], dtype=torch.long).to(self.device)

            # Cuts
            cuts_tensor, valid_mask = self._prepare_cuts(master)
            cuts_tensor = cuts_tensor.unsqueeze(0).to(self.device)
            valid_mask = valid_mask.unsqueeze(0).to(self.device)

            # 2. 模型推理
            predictions = self.model(
                cuts=cuts_tensor,
                valid_mask=valid_mask,
                lambda_=lambda_tensor,
                x_prev=x_prev_tensor,
                realization=realization_tensor,
                stage=stage_tensor,
            )

            # 3. 解析输出
            subgradient = predictions["subgradient"].squeeze(0).cpu().numpy()
            opt_value = predictions["opt_value"].squeeze(0).cpu().numpy().item()

        return subgradient, opt_value

    def _solve_with_warm_start(
        self,
        pi: np.ndarray,
        time_limit: float = None,
        master: MasterProblem = None,
    ):
        """
        用 ML 提供初始值，再用 Gurobi 求解

        TODO: 实现
        """
        # 先用 ML 预测初始值
        # 然后设置 Gurobi 的初始值
        # 最后调用 Gurobi 求解
        logger.warning("warm_start mode not implemented yet, falling back to original solve")
        return super().solve(pi, time_limit)

    def _flatten_trial_point(self, trial_point):
        """
        将 trial_point (x, y, x_bs, soc) 展开成单个向量
        """
        x, y, x_bs, soc = trial_point

        # 展开 x_bs
        x_bs_flat = [item for sublist in x_bs for item in sublist]

        # 拼接所有部分
        flat = np.concatenate([x, y, x_bs_flat, soc])
        return flat

    def _get_realization_features(self):
        """
        获取 realization 特征 (demand, renewable generation)

        TODO: 从 problem_params 中获取真实的 realization 数据
        """
        # 暂时返回占位值
        return np.array([0.0, 0.0])

    def _prepare_cuts(self, master: MasterProblem = None):
        """
        准备 cuts 数据

        Args:
            master: MasterProblem 对象

        Returns:
            cuts_tensor: shape (max_cuts, cut_dim)
            valid_mask: shape (max_cuts,), True 表示有效
        """
        max_cuts = self.max_cuts
        cut_dim = self.ml_config.dimensions["cut_dim"]

        # 初始化
        cuts = np.zeros((max_cuts, cut_dim), dtype=np.float32)
        valid_mask = np.zeros(max_cuts, dtype=bool)

        if master is not None:
            all_cuts = master.get_all_cuts()
            num_cuts = min(len(all_cuts), max_cuts)

            for i in range(num_cuts):
                g_new, x_new, f_new = all_cuts[i]

                # 构建 cut 特征: g_i + phi_i
                # phi_i = f_new - g_new @ x_new
                phi_i = f_new - np.dot(g_new, x_new)

                # 拼接
                cut_feature = np.concatenate([g_new, [phi_i]])

                # 填充
                cuts[i, : len(cut_feature)] = cut_feature
                valid_mask[i] = True

        return torch.tensor(cuts, dtype=torch.float32), torch.tensor(valid_mask, dtype=torch.bool)
