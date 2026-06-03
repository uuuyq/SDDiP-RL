# RL 预测步长 eta
说明：由于只需要预测eta，因此后面还是需要使用eta求解得出新的pi值

求解master可以得到上界，sub problem可以得到下界，因此还是可以得出gap，具体的计算可以参考 D:\tools\workspace_pycharm\SDDiP-RL\bundle_RL\script\lag_problem.py

## 算法流程
1. RL 预测 eta
2. 将 eta 作为参数送入 master 问题求解
3. 得到 d 以及新的 pi 值
4. 求解子问题得到 cut
5. 更新 state，进入下一个阶段

Observation:
--------------------------------
obs = {
    # 当前η
    log_eta: log(eta),

    # 当前gap
    log_delta: log(delta),

    # serious/null  上一步是否成功改善center
    serious_step: 01值,

    # search direction
    d_norm_sq: ||d||²,  通过subproblem求解得到的pi的增量d 二范数

    # proximal term
    proximal_term: eta*||d||²,    master中二次项的大小

    # linearization error
    lin_error: linearization_error = (f_new + np.array(subgradient).dot(np.array(x_best) - np.array(x_new)) - f_best),

    # 最近收敛趋势
    gap_improve_1: log(current_gap) - log(prev_gap), 最后一步的gap改善，使用对数值减法
    gap_improve_2: log(current_gap) - log(prev_prev_gap), 最近两步的gap改善，使用对数值减法
    gap_improve_3: log(current_gap) - log(prev_prev_prev_gap), 最近三步的gap改善，使用对数值减法

    # bundle geometry
    g_norm_sq: ||g_t||²,  新的梯度大小
    cos_gd: cos(g_t, d)   余弦相似度
}
--------------------------------

Action:
--------------------------------
a ∈ [-1,1]
--------------------------------

Update:
--------------------------------
η_new = η_old * exp(0.5*a)
--------------------------------

Reward:
--------------------------------
与 default_feature 相同的奖励
--------------------------------

## 状态维护
需要维护：
- x_best: 最优 pi
- f_best: 最优 phi
- i_u: weight update 计数器
- cuts: 历史 cuts
- gap_history: 用于计算 gap_improve_1/2/3
