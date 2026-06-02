import time

import numpy as np
from bundle_RL.script.lag_problem import SubProblem, MasterProblem


def bundle_baseline(logger, config, tolerance=1e-5):
    """传统 Bundle 算法求解作为 baseline"""
    sub = SubProblem(logger, config, n=config.n)
    master = MasterProblem(logger, config.N_VARS, tolerance=tolerance)
    
    delta_history = []
    time_history = []
    ub_history = []      # 上界历史
    f_best_history = []  # 最优下界历史
    x_new = np.zeros(config.N_VARS)
    g_new, f_new = sub.solve(x_new)
    master.update_strategy(x_new, f_new, g_new, ub=None)
    f_best_history.append(master.f_best)
    
    for i in range(20):
        start_time = time.time()
        master.add_cut(x_new, f_new, g_new)
        ub, x_new = master.solve_master()
        g_new, f_new = sub.solve(x_new)
        serious_step, delta, stop_flag = master.update_strategy(x_new, f_new, g_new, ub)
        end_time = time.time()
        
        delta_history.append(delta)
        time_history.append(end_time - start_time)
        ub_history.append(ub)
        f_best_history.append(master.f_best)
        logger.info(f"Baseline - rel_gap: {delta:.6e}, time: {time_history[-1]:.4f}s")
        if stop_flag:
            break
    
    return delta_history, time_history, ub_history, f_best_history


def bundle_RL(env, model, master, logger, deterministic):
    delta_history = []
    reward_history = []
    time_history = []
    ub_history = []      # 上界历史
    f_best_history = []  # 最优下界历史

    obs, _ = env.reset()

    sub_result = env.bundle[-1]
    x_new = sub_result["pi"]
    f_new = sub_result["phi"]
    g_new = sub_result["g"]
    # 第一次，更新 f_best和x_best
    master.update_strategy(x_new, f_new, g_new, ub=None)
    f_best_history.append(master.f_best)

    logger.info("==== ROLLOUT ====")
    for step in range(20):
        start_time = time.time()
        master.add_cut(x_new, f_new, g_new)
        ub, x_new_bundle = master.solve_master()
        _, delta, stop_flag = master.update_strategy(x_new, f_new, g_new, ub=ub)
        action, _ = model.predict(obs, deterministic=deterministic)
        state, reward, terminated, truncated, info = env.step(action)
        end_time = time.time()
        
        # 保存数据
        delta_history.append(delta)
        reward_history.append(reward)
        time_history.append(end_time - start_time)
        ub_history.append(ub)
        f_best_history.append(master.f_best)
        logger.info(f"RL - rel_gap: {delta:.6e}, reward: {reward:.6e}, time: {time_history[-1]:.4f}s")
        
        # 获取新的子问题得到的cut
        sub_result = env.bundle[-1]
        x_new = sub_result["pi"]
        f_new = sub_result["phi"]
        g_new = sub_result["g"]

        # logger.info(f"############################")
        # logger.info(f"x_new_bundle: {x_new_bundle}   x_new_RL: {x_new}")
        # logger.info(f"############################")

        # 添加终止条件（与baseline保持一致）
        if stop_flag:
            logger.info(f"RL Model - 满足终止条件，提前停止，rel_gap: {delta:.6e}")
            break

    logger.info("Test finished successfully.")
    
    return delta_history, reward_history, time_history, ub_history, f_best_history


def bundle_RL_warmstart(env, model, master, logger, warmstart_threshold=1e-6, patience=3, deterministic=True):
    """
    Warm-start 测试方法：当 RL 的 delta 不再发生变化时，切换成 baseline 的计算方式
    
    Args:
        env: 环境
        model: RL 模型
        master: MasterProblem 对象
        logger: 日志记录器
        warmstart_threshold: delta 变化阈值，小于此值认为不再变化
        patience: 连续多少次 delta 变化小于阈值后切换到 baseline
    
    Returns:
        delta_history: delta 历史记录
        reward_history: 奖励历史记录（仅RL阶段）
        time_history: 时间历史记录
        ub_history: 上界历史
        f_best_history: 最优下界历史
        switch_step: 切换到 baseline 的步骤（None表示未切换）
    """
    delta_history = []
    reward_history = []
    time_history = []
    ub_history = []      # 上界历史
    f_best_history = []  # 最优下界历史
    switch_step = None
    consecutive_small_changes = 0
    
    obs, _ = env.reset()

    sub_result = env.bundle[-1]
    x_new = sub_result["pi"]
    f_new = sub_result["phi"]
    g_new = sub_result["g"]
    master.update_strategy(x_new, f_new, g_new, ub=None)
    f_best_history.append(master.f_best)

    logger.info("==== WARMSTART ROLLOUT ====")
    
    # RL阶段
    for step in range(20):
        start_time = time.time()
        master.add_cut(x_new, f_new, g_new)
        ub, _ = master.solve_master()
        _, delta, stop_flag = master.update_strategy(x_new, f_new, g_new, ub=ub)
        action, _ = model.predict(obs, deterministic=deterministic)
        state, reward, terminated, truncated, info = env.step(action)
        end_time = time.time()
        
        delta_history.append(delta)
        reward_history.append(reward)
        time_history.append(end_time - start_time)
        ub_history.append(ub)
        f_best_history.append(master.f_best)
        logger.info(f"[RL] rel_gap: {delta:.6e}, reward: {reward:.6e}, time: {time_history[-1]:.4f}s")
        
        sub_result = env.bundle[-1]
        x_new = sub_result["pi"]
        f_new = sub_result["phi"]
        g_new = sub_result["g"]
        
        # 检查是否满足终止条件
        if stop_flag:
            logger.info(f"Warmstart - RL阶段满足终止条件，rel_gap: {delta:.6e}")
            return delta_history, reward_history, time_history, ub_history, f_best_history, switch_step
        
        # 检查 rel_gap 是否不再变化或反向上升（用于判断是否切换到 baseline）
        if len(delta_history) >= 2:
            rel_gap_change = abs(delta_history[-1] - delta_history[-2])
            rel_gap_ref = max(abs(delta_history[-1]), 1)
            rel_gap_relative_change = rel_gap_change / rel_gap_ref
            
            # 判断条件：gap反向上升 OR 下降不明显
            gap_increased = delta_history[-1] > delta_history[-2]
            gap_not_decreasing = rel_gap_relative_change < warmstart_threshold
            
            if gap_increased:
                # 情况1：rel_gap反向上升，立即退出RL模式
                logger.info(f"Warmstart - rel_gap反向上升: {delta_history[-2]:.6e} -> {delta_history[-1]:.6e}，立即切换到baseline模式")
                switch_step = step + 1
                # 使用 env.bundle[-2] 的结果，因为 bundle[-1] 的结果不好（导致gap上升）
                if len(env.bundle) >= 2:
                    sub_result = env.bundle[-2]
                    x_new = sub_result["pi"]
                    f_new = sub_result["phi"]
                    g_new = sub_result["g"]
                    logger.info(f"Warmstart - 使用 bundle[-2] 的结果作为 baseline 起点")
                break
            elif gap_not_decreasing:
                # 情况2：rel_gap下降不明显，累计计数
                consecutive_small_changes += 1
                logger.info(f"Warmstart - rel_gap相对变化: {rel_gap_relative_change:.6e}, 连续次数: {consecutive_small_changes}/{patience}")
                
                if consecutive_small_changes >= patience:
                    logger.info(f"Warmstart - rel_gap连续{patience}次下降不明显，切换到baseline模式")
                    switch_step = step + 1
                    # 使用当前bundle[-1]的结果（只是变化慢，不是变差）
                    sub_result = env.bundle[-1]
                    x_new = sub_result["pi"]
                    f_new = sub_result["phi"]
                    g_new = sub_result["g"]
                    logger.info(f"Warmstart - 使用当前bundle[-1]的结果作为 baseline 起点")
                    break
            else:
                consecutive_small_changes = 0
    
    # 如果切换到 baseline 模式
    if switch_step is not None:
        logger.info("==== SWITCHING TO BASELINE ====")
        for step in range(20 - switch_step):
            start_time = time.time()
            master.add_cut(x_new, f_new, g_new)
            ub, x_new = master.solve_master()
            g_new, f_new = env.subproblem.solve(x_new)
            serious_step, delta, stop_flag = master.update_strategy(x_new, f_new, g_new, ub)
            end_time = time.time()
            
            delta_history.append(delta)
            time_history.append(end_time - start_time)
            ub_history.append(ub)
            f_best_history.append(master.f_best)
            logger.info(f"[Baseline] rel_gap: {delta:.6e}, time: {time_history[-1]:.4f}s")
            
            if stop_flag:
                logger.info(f"Warmstart - Baseline阶段满足终止条件，rel_gap: {delta:.6e}")
                break

    logger.info("Warmstart test finished successfully.")
    
    return delta_history, reward_history, time_history, ub_history, f_best_history, switch_step