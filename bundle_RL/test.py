from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from train import train
from bundle_RL.lag_problem import SubProblem, MasterProblem
from bundle_RL.logger import get_logger
from bundle_env import BundleDualEnv, ProblemData
from sddip.sddip import parameters

def create_env(logger):
    # ===============================
    # 1️⃣ 构造 ProblemData
    # ===============================

    t = 0
    k = 0
    n = 0
    i = 0
    n_vars = 13
    x_trial = [1.0, 1.0, 1.0]
    y_trial = [71.52627531002818, 59.02627531002818, 66.52627531002818]
    x_bs_trial = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    soc_trial = [5.0]
    trial_point = (x_trial, y_trial, x_bs_trial, soc_trial)
    path = Path(r"D:\tools\workspace_pycharm\sddip-main-zou\data\01_test_cases\case6ww\t06_n06")
    problem_params = parameters.Parameters(path)

    problem_data = ProblemData(
        logger,
        problem_params,
        trial_point,
        t,
        n,
        i
    )
    master = MasterProblem(logger, n_vars, tolerance=1e-5)

    # ===============================
    # 2️⃣ 创建环境
    # ===============================

    state_dim = 13
    K = 20

    env = BundleDualEnv(
        problemData=problem_data,
        state_dim=state_dim,
        K=K
    )
    return env, master


def test(env, model, master, logger):

    obs, _ = env.reset()

    sub_result = env.bundle[-1]
    x_new = sub_result["pi"]
    f_new = sub_result["phi"]
    g_new = sub_result["g"]

    master.add_cut(x_new, f_new, g_new)

    master.update_strategy(x_new, f_new, g_new)

    master.solve()
    logger.info(f"上界 f_hat: {master.f_hat}")


    logger.info("==== ROLLOUT ====")

    for step in range(20):
        action, _ = model.predict(obs, deterministic=True)
        state, reward, terminated, truncated, info = env.step(action)

        logger.info(f"Step {step + 1}")
        # logger.info(f"  pi norm: {np.linalg.norm(state['pi'])}")
        # logger.info(f"  cuts: {state['cuts']}")
        # logger.info(f"  pi: {state['pi']}")
        logger.info(f"  reward: {reward}")
        logger.info(f"  bundle size: {len(env.bundle)}")
        logger.info("")

    logger.info("Test finished successfully.")


def main():
    logger = get_logger("log/bundle_env_test.log")
    env, master = create_env(logger)

    # model = train(env)

    # 加载model
    model = PPO.load("model/ppo_bundle_0312_1633.zip")

    test(env, model, logger)


if __name__ == "__main__":
    main()



"""
--------------------------------------------------
[Rollout 阶段 - 业务表现]
- ep_rew_mean: 
    含义: 回合平均总奖励。
    判断: 核心指标，必须长期看涨。如果不涨，检查 Reward 函数。
- ep_len_mean: 
    含义: 回合平均长度。
    判断: 判定模型是“早死”还是“通关”。

[Train 阶段 - 模型稳定性]
- entropy_loss: 
    含义: 策略熵（动作随机性）。
    判断: 绝对值应缓慢下降。绝对值过快趋近0表示过早收敛（不再尝试新动作）；
         一直很大表示模型在乱撞，学不到规律。
- explained_variance: 
    含义: 预测奖励的解释方差。
    判断: 越接近 1.0 越好。如果小于 0，说明 Critic 网络预测得比瞎猜还差。
- approx_kl: 
    含义: 新旧策略的 KL 散度（策略更新步长）。
    判断: 理想在 0.001 到 0.05 之间。若过大（如 >0.1），训练易崩溃。
- clip_fraction: 
    含义: 触发 PPO 截断机制的比例。
    判断: 常用 0.1~0.2。如果过高，说明更新被频繁强制限制。
- value_loss: 
    含义: 价值函数误差。
    判断: 代表评价员准不准，通常先升后降。

[Time 阶段 - 性能]
- fps: 
    含义: 每秒处理步数。
    判断: 衡量环境执行速度，主要受 env.step() 的复杂度影响。
"""