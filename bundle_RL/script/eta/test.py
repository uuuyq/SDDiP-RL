
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


def bundle_RL(env, model, logger, deterministic):
    """
    Eta 模块专用 RL 测试函数
    
    Args:
        env: eta 模块的 BundleDualEnv
        model: RL 模型
        logger: 日志记录器
        deterministic: 是否使用确定性策略
    
    Returns:
        delta_history: delta 历史记录
        reward_history: 奖励历史记录
        time_history: 时间历史记录
        ub_history: 上界历史记录
        f_best_history: 最优下界历史记录
    """
    delta_history = []
    reward_history = []
    time_history = []
    ub_history = []      # 上界历史
    f_best_history = []  # 最优下界历史

    obs, _ = env.reset()

    # 第一次，记录初始状态
    f_best_history.append(env.f_best)

    logger.info("==== ROLLOUT ====")
    for step in range(env.K):
        start_time = time.time()
        
        action, _ = model.predict(obs, deterministic=deterministic)
        state, reward, terminated, truncated, info = env.step(action)
        
        obs = state

        end_time = time.time()
        
        # 保存数据
        delta_history.append(env.delta)
        reward_history.append(reward)
        time_history.append(end_time - start_time)
        
        # ub 是上一次求解的 master 目标
        # 注意：eta 模块的 env 没有直接暴露 ub，我们可以使用 0 或者不保存
        # 这里我们用 0 占位，主要关注 f_best
        ub_history.append(0.0)
        f_best_history.append(env.f_best)
        
        logger.info(f"RL - rel_gap: {env.delta:.6e}, reward: {reward:.6e}, time: {time_history[-1]:.4f}s")
        
        if terminated:
            logger.info(f"RL Model - 达到最大迭代次数，提前停止")
            break

    logger.info("Test finished successfully.")
    
    return delta_history, reward_history, time_history, ub_history, f_best_history


def bundle_RL_warmstart(env, model, logger, warmstart_threshold=1e-6, patience=3, deterministic=True):
    """
    Warm-start 测试方法（Eta 模块专用）
    
    Args:
        env: eta 模块的 BundleDualEnv
        model: RL 模型
        logger: 日志记录器
        warmstart_threshold: delta 变化阈值
        patience: 连续多少次 delta 变化小于阈值后切换到 baseline
    
    Returns:
        delta_history: delta 历史记录
        reward_history: 奖励历史记录（仅RL阶段）
        time_history: 时间历史记录
        ub_history: 上界历史记录
        f_best_history: 最优下界历史记录
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
    f_best_history.append(env.f_best)

    logger.info("==== WARMSTART ROLLOUT ====")
    
    # RL阶段
    for step in range(env.K):
        start_time = time.time()
        
        action, _ = model.predict(obs, deterministic=deterministic)
        state, reward, terminated, truncated, info = env.step(action)
        
        obs = state

        end_time = time.time()
        
        delta_history.append(env.delta)
        reward_history.append(reward)
        time_history.append(end_time - start_time)
        ub_history.append(0.0)
        f_best_history.append(env.f_best)
        logger.info(f"[RL] rel_gap: {env.delta:.6e}, reward: {reward:.6e}, time: {time_history[-1]:.4f}s")
        
        # 检查是否满足终止条件
        if terminated:
            logger.info(f"Warmstart - RL阶段达到最大迭代次数")
            return delta_history, reward_history, time_history, ub_history, f_best_history, switch_step
        
        # 检查 rel_gap 是否不再变化或反向上升
        if len(delta_history) >= 2:
            rel_gap_change = abs(delta_history[-1] - delta_history[-2])
            rel_gap_ref = max(abs(delta_history[-1]), 1)
            rel_gap_relative_change = rel_gap_change / rel_gap_ref
            
            gap_increased = delta_history[-1] > delta_history[-2]
            gap_not_decreasing = rel_gap_relative_change < warmstart_threshold
            
            if gap_increased:
                logger.info(f"Warmstart - rel_gap反向上升: {delta_history[-2]:.6e} -> {delta_history[-1]:.6e}，立即切换到baseline模式")
                switch_step = step + 1
                break
            elif gap_not_decreasing:
                consecutive_small_changes += 1
                logger.info(f"Warmstart - rel_gap相对变化: {rel_gap_relative_change:.6e}, 连续次数: {consecutive_small_changes}/{patience}")
                
                if consecutive_small_changes >= patience:
                    logger.info(f"Warmstart - rel_gap连续{patience}次下降不明显，切换到baseline模式")
                    switch_step = step + 1
                    break
            else:
                consecutive_small_changes = 0
    
    # 如果切换到 baseline 模式
    if switch_step is not None:
        logger.info("==== SWITCHING TO BASELINE ====")
        
        # 创建一个新的 MasterProblem，用当前 env 的状态初始化
        master = MasterProblem(logger, env.state_dim, tolerance=1e-5)
        sub = SubProblem(logger, env.config, n=env.n)
        
        # 用 env 中的 cuts 初始化 master
        for g, pi, f in env.cuts_storage:
            master.add_cut(pi, f, g)
        
        # 初始化 master 的状态
        master.x_best = env.x_best.copy()
        master.f_best = env.f_best
        master.iter_idx = len(env.cuts_storage)
        
        # 用当前的 x_new 和 f_new 作为起点
        x_new = env.x_new.copy()
        f_new = env.f_new
        g_new = env.g_new
        
        # 继续用 baseline 方式迭代
        remaining_steps = 20 - switch_step
        for step in range(remaining_steps):
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
            logger.info(f"[Baseline] rel_gap: {delta:.6e}, time: {time_history[-1]:.4f}s")
            
            if stop_flag:
                logger.info(f"Warmstart - Baseline阶段满足终止条件，rel_gap: {delta:.6e}")
                break

    logger.info("Warmstart test finished successfully.")
    
    return delta_history, reward_history, time_history, ub_history, f_best_history, switch_step

