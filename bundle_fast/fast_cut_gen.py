"""
fast_cut_gen.py

分批次使用 subgradients 和 mu_weights 计算 pi，
然后使用每个 pi 求解子问题得到 cut，保存到 JSON。

pi 计算方式：
    从 0 开始，每次增加 step 个 g
    pi_k = sum_{j=0}^{k} (mu[j] / sum(mu[0:k+1])) * g[j]

cut 结构（与 lag_problem.py 一致）：
    {"g": [...], "x": [...], "f": value}
"""

import json

import numpy as np

from bundle_fast.lag_problem import SubProblem
from bundle_fast.logger import get_logger
from bundle_fast import config


def load_subgradients_mu(file_path: str) -> tuple[list, list]:
    """加载 subgradients 和 mu_weights"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["subgradients"], data["mu_weights"]


def compute_pi_list(subgradients: list, mu_weights: list, step: int = 1) -> list:
    """
    计算分批次的 pi 列表

    累积方式：每次增加 step 个 subgradient
    - step=1: 第1次用g[0], 第2次用g[0:2], 第3次用g[0:3], ...
    - step=2: 第1次用g[0:2], 第2次用g[0:4], 第3次用g[0:6], ...

    Args:
        subgradients: 梯度列表，每个元素是一个数组
        mu_weights: 权重列表
        step: 每次增加的梯度数量

    Returns:
        pi_list: 累积加权的 pi 值列表
    """
    n = len(subgradients)
    pi_list = []

    # 累积使用 subgradient
    for end_idx in range(step, n + 1, step):
        # 使用 g[0:end_idx]，即从第0个到第 end_idx-1 个
        batch_mu = mu_weights[0:end_idx]

        # 归一化 mu
        sum_mu = sum(batch_mu)
        if sum_mu > 1e-12:
            normalized_mu = [m / sum_mu for m in batch_mu]
        else:
            # 如果 mu 全为 0，平均分配
            normalized_mu = [1.0 / len(batch_mu)] * len(batch_mu)

        # 计算累积加权的 pi
        pi = np.zeros_like(np.array(subgradients[0]))
        for j in range(end_idx):
            pi += normalized_mu[j] * np.array(subgradients[j])

        pi_list.append(pi.tolist())

    return pi_list


def solve_subproblem_for_cut(
    logger, problem_params, trial_point, t, n, i, pi
) -> dict:
    """
    使用给定的 pi 求解子问题，返回 cut

    Returns:
        {"g": [...], "x": [...], "f": value}
    """
    sub = SubProblem(logger, problem_params, trial_point, t, n, i)
    g, f = sub.solve(pi)

    return {
        "g": g.tolist(),
        "x": pi.tolist(),
        "f": float(f)
    }


def generate_cuts(
    subgradients: list,
    mu_weights: list,
    step: int = 1,
    t: int = None,
    realization: int = 1,
    trial_point=None,
    problem_params=None,
    logger=None,
) -> list:
    """
    生成分批次的 cuts

    Args:
        subgradients: 梯度列表
        mu_weights: 权重列表
        step: 每次增加的梯度数量
        t: 时间阶段
        realization: 场景索引
        trial_point: 试探点
        problem_params: 问题参数
        logger: 日志器

    Returns:
        cuts: cut 列表
    """
    if trial_point is None:
        trial_point = (config.X_TRIAL, config.Y_TRIAL, config.X_BS_TRIAL, config.SOC_TRIAL)

    if problem_params is None:
        problem_params = config.PROBLEM_PARAMS

    # 计算 pi 列表
    pi_list = compute_pi_list(subgradients, mu_weights, step)

    logger.info(f"将生成 {len(pi_list)} 个 cuts (step={step})")

    # 求解子问题生成 cuts
    cuts = []
    for idx, pi in enumerate(pi_list):
        logger.info(f"求解 cut {idx + 1}/{len(pi_list)}: pi = {pi}")
        cut = solve_subproblem_for_cut(
            logger, problem_params, trial_point, t, realization, 0, np.array(pi)
        )
        cuts.append(cut)
        logger.info(f"cut {idx + 1}: f = {cut['f']}")

    return cuts


def save_cuts(cuts: list, output_file: str):
    """保存 cuts 到 JSON 文件"""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cuts, f, ensure_ascii=False, indent=4)
    print(f"已保存 {len(cuts)} 个 cuts 到 {output_file}")


def main(
    input_file: str = "fast_g_gen/subgradients_mu.json",
    output_file: str = "cuts_fast.json",
    step: int = 1,
    t: int = None,
    realization: int = 1,
):
    """
    主函数

    Args:
        input_file: 输入的 subgradients 和 mu 文件
        output_file: 输出的 cuts 文件
        step: 每次增加的梯度数量
        t: 时间阶段
        realization: 场景索引
    """
    if t is None:
        t = config.T
    logger = get_logger("log/fast_cut_gen.log")

    # 加载数据
    subgradients, mu_weights = load_subgradients_mu(input_file)
    logger.info(f"加载了 {len(subgradients)} 个 subgradients 和 {len(mu_weights)} 个 mu_weights")

    # 生成 cuts
    cuts = generate_cuts(
        subgradients, mu_weights, step, t, realization, logger=logger
    )

    # 保存 cuts
    save_cuts(cuts, output_file)

    return cuts


if __name__ == "__main__":
    # 在这里直接设置参数
    main(
        input_file="fast_g_gen/subgradients_mu.json",
        output_file="fast_cut_gen/cuts_fast.json",
        step=1,
        t=config.T,
        realization=1,
    )
