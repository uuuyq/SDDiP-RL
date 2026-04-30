"""
fast_cut_gen.py

分批次使用 subgradients 和 mu_weights 计算 pi，
然后使用每个 pi 求解子问题得到 cut。

pi 计算方式：
    从 0 开始，每次增加 step 个 g
    pi_k = sum_{j=0}^{k} (mu[j] / sum(mu[0:k+1])) * g[j]

cut 结构（与 lag_problem.py 一致）：
    {"g": [...], "x": [...], "f": value}
"""

import json

import numpy as np

from bundle_fast.lag_problem import SubProblem
from bundle_fast.config import BundleConfig


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

    for end_idx in range(step, n + 1, step):
        batch_mu = mu_weights[0:end_idx]

        sum_mu = sum(batch_mu)
        if sum_mu > 1e-12:
            normalized_mu = [m / sum_mu for m in batch_mu]
        else:
            normalized_mu = [1.0 / len(batch_mu)] * len(batch_mu)

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
    config: BundleConfig,
    subgradients: list,
    mu_weights: list,
    step: int = 1,
    realization: int = 1,
    logger=None,
) -> list:
    """
    生成分批次的 cuts

    Args:
        config: BundleConfig 配置对象
        subgradients: 梯度列表
        mu_weights: 权重列表
        step: 每次增加的梯度数量
        realization: 场景索引
        logger: 日志器

    Returns:
        cuts: cut 列表
    """
    trial_point = config.trial_point
    problem_params = config.PROBLEM_PARAMS
    t = config.T

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
        logger.info(f"cut {idx + 1}: f = {cut['f']:.6f}")

    return cuts


def save_cuts(cuts: list, output_file: str, logger=None):
    """保存 cuts 到 JSON 文件"""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cuts, f, ensure_ascii=False, indent=4)
    if logger:
        logger.info(f"已保存 {len(cuts)} 个 cuts 到 {output_file}")
    print(f"已保存 {len(cuts)} 个 cuts 到 {output_file}")


def compute_objective_terms_value(x_vars: dict, problem_params) -> float:
    """
    计算 objective_terms 的值

    objective_terms = sum(coefficients[i] * variables[i])

    在 problem_params 中，cost_coeffs 已经包含：
        gc + suc + sdc + [penalty] * 2

    扩展后的 coefficients = cost_coeffs + [penalty] * (2*n_storages + 2*n_x_bs + 1) + [1]
    variables = y + s_up + s_down + [ys_p, ys_n] + socs_p + socs_n + x_bs_p + x_bs_n + [delta] + [theta]

    Args:
        x_vars: 包含所有变量值的字典
        problem_params: 问题参数对象，包含 cost_coeffs 和 penalty

    Returns:
        objective_terms_value: objective_terms 的值
    """
    # 获取系数（cost_coeffs 已经包含 gc + suc + sdc + [penalty] * 2）
    cost_coeffs = problem_params.cost_coeffs
    penalty = problem_params.penalty

    n_generators = problem_params.n_gens
    n_storages = problem_params.n_storages
    backsight_periods = problem_params.backsight_periods

    # 计算 x_bs_p 和 x_bs_n 的总数
    n_x_bs = sum(backsight_periods)

    # 构建完整的系数列表
    # cost_coeffs 已经包含 gc + suc + sdc + [penalty] * 2
    # 扩展后的 coefficients = cost_coeffs + [penalty] * (2*n_storages + 2*n_x_bs + 1)
    coefficients = cost_coeffs + [penalty] * (2 * n_storages + 2 * n_x_bs + 1)

    # 构建变量值列表
    # variables = y + s_up + s_down + [ys_p, ys_n] + socs_p + socs_n + x_bs_p + x_bs_n + [delta] + [theta]

    # y: n_generators 个
    y_values = x_vars['y']

    # s_up: n_generators 个
    s_up_values = x_vars['s_up']

    # s_down: n_generators 个
    s_down_values = x_vars['s_down']

    # ys_p, ys_n: 2 个全局变量
    ys_p_value = x_vars['ys_p']
    ys_n_value = x_vars['ys_n']

    # socs_p, socs_n: n_storages 个
    socs_p_values = x_vars['socs_p']
    socs_n_values = x_vars['socs_n']

    # x_bs_p, x_bs_n: n_generators 个（每个发电机一个）
    x_bs_p_values = x_vars['x_bs_p']
    x_bs_n_values = x_vars['x_bs_n']

    # delta: 1 个
    delta_value = x_vars['delta']

    # theta: 1 个
    theta_value = x_vars['theta']

    # 组合变量值列表
    variables_values = (
        y_values + s_up_values + s_down_values
        + [ys_p_value, ys_n_value]
        + socs_p_values + socs_n_values
        + x_bs_p_values + x_bs_n_values
        + [delta_value, theta_value]
    )

    # 计算 objective_terms_value
    objective_terms_value = sum(c * v for c, v in zip(coefficients, variables_values))

    return objective_terms_value


def generate_cuts_update(
    config: BundleConfig,
    subgradients: list,
    mu_weights: list,
    x_vars_list: list,
    step: int = 1,
    logger=None,
) -> list:
    """
    使用已知数据直接计算 cuts，不求解子问题

    对于每个累积加权的 pi，使用对应的 subgradient 和 x_vars 直接计算截距 f。

    截距计算公式：
    - objective_terms_value = sum(coefficients[i] * variables[i])
    - f = objective_terms_value

    Args:
        config: BundleConfig 配置对象
        subgradients: 梯度列表
        mu_weights: 权重列表
        x_vars_list: 每个组的 x 变量值列表
        step: 每次增加的梯度数量
        logger: 日志器

    Returns:
        cuts: cut 列表
    """
    problem_params = config.PROBLEM_PARAMS

    # 使用 compute_pi_list 计算 pi 列表
    pi_list = compute_pi_list(subgradients, mu_weights, step)

    logger.info(f"将生成 {len(pi_list)} 个 cuts (step={step}, 使用直接计算方式)")

    # 直接计算 cuts
    cuts = []
    for idx, pi in enumerate(pi_list):
        end_idx = step * (idx + 1)

        # 计算加权平均的 x_vars（使用与 compute_pi_list 相同的权重）
        batch_mu = mu_weights[0:end_idx]
        sum_mu = sum(batch_mu)
        if sum_mu > 1e-12:
            normalized_mu = [m / sum_mu for m in batch_mu]
        else:
            normalized_mu = [1.0 / len(batch_mu)] * len(batch_mu)

        # 计算加权平均的 x_vars
        weighted_x_vars = {}
        for key in x_vars_list[0].keys():
            values = [x_vars_list[j][key] for j in range(end_idx)]
            if isinstance(values[0], list):
                weighted_value = [
                    sum(normalized_mu[j] * values[j][i] for j in range(end_idx))
                    for i in range(len(values[0]))
                ]
                weighted_x_vars[key] = weighted_value
            else:
                weighted_x_vars[key] = sum(normalized_mu[j] * values[j] for j in range(end_idx))

        # 计算 objective_terms_value 作为截距 f
        f = compute_objective_terms_value(weighted_x_vars, problem_params)

        # g 使用 pi（累积加权的 subgradient）
        g = np.array(pi)

        cut = {
            "g": g.tolist(),
            "x": pi,
            "f": float(f)
        }
        cuts.append(cut)
        logger.info(f"cut {idx + 1}: f = {f:.6f}")

    return cuts


if __name__ == "__main__":
    from bundle_fast.logger import get_logger
    from bundle_fast.config import get_default_config

    logger = get_logger("log/fast_cut_gen.log")
    config = get_default_config()

    # 加载数据
    input_file = "fast_g_gen/subgradients_mu.json"
    subgradients, mu_weights = load_subgradients_mu(input_file)
    logger.info(f"加载了 {len(subgradients)} 个 subgradients 和 {len(mu_weights)} 个 mu_weights")

    # 生成 cuts
    cuts = generate_cuts(
        config=config,
        subgradients=subgradients,
        mu_weights=mu_weights,
        step=1,
        realization=1,
        logger=logger,
    )

    # 保存 cuts
    output_file = "fast_cut_gen/cuts_fast.json"
    save_cuts(cuts, output_file, logger)
