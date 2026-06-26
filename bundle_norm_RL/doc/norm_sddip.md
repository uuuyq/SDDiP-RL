\documentclass[UTF8]{ctexart}

\title{norm sddip}
\author{Q}
\date{2026}

         
\usepackage{tikz}
\usepackage{amsmath, amssymb, amsfonts}   
\usepackage{graphicx}           
\usepackage{geometry}           
\usepackage{tikz}
\usepackage{booktabs} % 引入三线表宏包
\usepackage{float}    % 引入强力浮动体控制

% 配置 TikZ 绘图所需的关键扩展库
\usetikzlibrary{positioning, shapes.geometric, arrows.meta}
\tikzset{>={Stealth[scale=1.1]}} % 全局美化 TikZ 箭头样式
\geometry{a4paper, margin=2.5cm} 


\begin{document}

\maketitle


\section{多阶段问题定义、状态解耦与 Epigraph 近似}

\subsection{原始多阶段随机规划值函数}
在第 $t$ 阶段，给定前一阶段状态变量 $x_{t-1}$ 和当前场景的随机向量 $\xi_t$，最优值函数 $Q_t(x_{t-1}, \xi_t)$ 定义为：
\begin{equation}
Q_{t}(x_{t-1}, \xi_{t}) := \min_{x_{t}, y_{t}} \left\{ f_{t}(x_{t}, y_{t}) + \sum_{k \in K} p_{k} Q_{t+1}(x_{t}, \xi_{t+1}^{k}) : (x_{t}, y_{t}) \in \Phi_{n}(x_{t-1}, \xi_{t}) \right\}
\end{equation}
其中 $\Phi_{n}(x_{t-1}, \xi_{t})$ 表示在给定入状态 $x_{t-1}$ 和随机实现 $\xi_t$ 下的可行域，$K$ 为下一阶段的场景集合，$p_k$ 为场景 $k$ 的概率。

\subsection{引入状态解耦变量 $z_t$}
为了消除前后阶段状态变量的直接耦合，引入复制变量 $z_{t} = x_{t-1}$。未来阶段的期望值函数通过上方图（Epigraph）集合来约束逼近，问题可调整为如下等价形式：
\begin{align}
Q_{t}(x_{t-1}, \xi_{t}) = \min_{x_{t}, y_{t}, \theta_{t+1}^{k}} \quad & f_{t}(x_{t}, y_{t}) + \sum_{k \in K} p_{k} \theta_{t+1}^{k} \\
\text{s.t.} \quad & (x_{t}, y_{t}) \in \Phi_{n}(z_{t}, \xi_{t}) \\
& z_{t} = x_{t-1} \\
& \theta_{t+1}^{k} \ge Q_{t+1}(x_{t}, \xi_{t+1}^{k}), \quad \forall k \in K \\
& z_{t} \in X_{t-1}, \quad x_{t} \in X_{t}
\end{align}
注意：$z_t$ 松弛为连续变量

\subsection{值函数的 Cut 逼近近似}
在迭代过程中，通过对未来所有场景依据概率 $p_k$ 取期望值，构造出与具体场景 $k$ 无关的统一状态逼近函数 $\psi_{t+1}^{i}(x_t)$。具体地，设前 $i$ 次迭代中已生成 $i$ 条仿射割平面，第 $j$ 条割对应的仿射函数定义为：
\begin{equation}
\lambda_{j}(x_t) := \theta^{j} + (\pi^{j})^\top x_t
\end{equation}
其中 $\theta^{j}$ 为割的截距，$\pi^{j}$ 为割的斜率向量（对偶乘子）。值函数的逼近取所有历史割的上包络：
\begin{equation}
\psi_{t+1}^{i}(x_t) = \max \left\{ \lambda_{1}(x_t), \lambda_{2}(x_t), \dots, \lambda_{i}(x_t) \right\}
\end{equation}
在第 $i+1$ 次迭代的前向过程中，用 $\psi_{t+1}^{i}(x_t)$ 替代真实的期望值函数，第 $t$ 阶段的近似子模型表述为：
\begin{align}
\mathcal{Q}_{t}^{i+1}(x_{t-1}, \xi_t) = \min_{x_t, y_t, \theta_{t+1}} \quad & f_{t}(x_{t}, y_{t}) + \theta_{t+1} \\
\text{s.t.} \quad & (x_{t}, y_{t}) \in \Phi_{n}(z_{t}, \xi_{t}) \\
& z_{t} = x_{t-1}^{i} \\
& \theta_{t+1} \ge \psi_{t+1}^{i}(x_{t}) \\
& z_{t} \in Z_{t-1}, \quad x_{t} \in X_{t}
\end{align}

\subsection{Epigraph 可行性判定子问题}
为了检验当前前向探索点 $(x_{t-1}^i, \theta_t^i)$ 是否满足包络约束，我们需要求解一个可行性判定子问题。若该点不满足当前的近似上方图，则通过下式寻找违解程度：
\begin{align}
\min_{x_t, y_t, z_t, \theta_{t+1}} \quad & 0 \\
\text{s.t.} \quad & f_t(x_t, y_t) + \theta_{t+1} \le \theta_{t} \\
& \theta_{t+1} \ge \psi_{t+1}^{i}(x_t) \\
& (x_{t}, y_{t}) \in \Phi_{n}(z_{t}, \xi_{t}) \\
& z_{t} = x_{t-1} \\
& z_{t} \in Z_{t-1}, \quad x_{t} \in X_{t}
\end{align}

\section{拉格朗日松弛与对偶问题的外层 Bundle 求解}

\subsection{可行性问题的拉格朗日松弛与对偶割（Lagrangian Cut）的推导}
对状态解耦约束 $z_t = x_{t-1}$（引入乘子 $\pi$）以及状态下界约束 $f_t(x_t, y_t) + \theta_{t+1} - \theta_t \le 0$（引入乘子 $\pi_0 \ge 0$）实施拉格朗日松弛。
外层拉格朗日对偶问题的目标为：
\begin{equation}
\max_{\pi, \pi_0 \ge 0} \quad \omega_t^{i+1}(\pi, \pi_0) - \pi^\top x_{t-1}^i - \pi_0 \theta_t^i
\end{equation}
其中，由于约束条件中的松弛系数同时作用于 $f_t$ 和 $\theta_{t+1}$，内层拉格朗日松弛子问题严格修正为：
\begin{align}
\omega_t^{i+1}(\pi, \pi_0) = \min_{x_t, y_t, z_t, \theta_{t+1}} \quad & \pi_0 f_t(x_t, y_t) + \pi^\top z_t + \pi_0 \theta_{t+1} \\
\text{s.t.} \quad & (x_{t}, y_{t}) \in \Phi_{n}(z_{t}, \xi_{t}) \\
& \theta_{t+1} \ge \psi_{t+1}^{i}(x_{t}) \\
& z_{t} \in Z_{t-1}, \quad x_{t} \in X_{t}
\end{align}

若当前点 $(x_{t-1}^i, \theta_t^i)$ 违背了可行性，根据对偶理论，其外层对偶问题的最优目标值严格大于 0：
\begin{equation}
\omega_t^{i+1}(\pi, \pi_0) - \pi^\top x_{t-1}^i - \pi_0 \theta_t^i > 0
\end{equation}
由此可构造出原问题的拉格朗日可行性割：
\begin{equation}
\pi^\top x_{t-1} + \pi_0 \theta_t \ge \omega_t^{i+1}(\pi, \pi_0)
\end{equation}
当选取 $\pi_0 > 0$ 时，经过归一化移项后，可以转化为关于 $\theta_t$ 的经典下界仿射割形式：
\begin{equation}
\theta_t \ge \frac{\omega_t^{i+1}(\pi, \pi_0)}{\pi_0} - \left(\frac{\pi}{\pi_0}\right)^\top x_{t-1}
\end{equation}

\subsection{ 基于 Bundle 方法的对偶迭代求解}
\subsubsection{对偶乘子初始化}
\textbf{$\pi_0$ 统一初始化}：
    控制值函数基础权重的对偶乘子 $\pi_0$ 均统一初始化为常数 $1.0$：
    \begin{equation}
    \pi_0^{(0)} = 1.0
    \end{equation}
\begin{itemize}
    \item \textbf{ZeroDuals（零向量初始化）}：
    直接将状态解耦对偶乘子 $\pi$ 初始化为全零向量：
    \begin{equation}
    \pi^{(0)} = \mathbf{0}
    \end{equation}

    \item \textbf{LPDuals（LP 松弛对偶初始化）}：
    该策略通过求解当前第 $t$ 阶段近似子模型的线性规划（LP）松弛问题来获取初始乘子。定义其连续松弛子模型为（即将原问题中的整数变量、非线性域等全部进行线性或凸松弛）：
    \begin{align}
    \min_{x_t, y_t, z_t, \theta_{t+1}} \quad & f_{t}(x_{t}, y_{t}) + \theta_{t+1} \\
    \text{s.t.} \quad & (x_{t}, y_{t}) \in \text{R-cl}\left(\Phi_{n}(z_{t}, \xi_{t})\right) \\
    & z_{t} = x_{t-1}^i \quad \left( \lambda \right) \\
    & \theta_{t+1} \ge \psi_{t+1}^{i}(x_{t}) \\
    & z_{t} \in \text{co}(Z_{t-1}), \quad x_{t} \in \text{co}(X_{t})
    \end{align}
    其中 $\text{R-cl}(\cdot)$ 和 $\text{co}(\cdot)$ 分别表示可行域与变量集合的连续/凸松弛约束。令该线性规划问题中状态解耦约束 $z_{t} = x_{t-1}^i$ 对应的最优对偶乘子（Shadow Price）为 $\lambda^*$。则 $\pi$ 的初始值赋值为：
    \begin{equation}
    \pi^{(0)} = \lambda^*
    \end{equation}

\end{itemize}

\subsubsection{Bundle算法}
外层对偶采用 Bundle（束方法）架构。在给定的内部循环中，将当前的中心乘子记为 $(\hat{\pi}, \hat{\pi}_0)$，代入内层松弛模型进行评估：
\begin{align}
\omega_t^{i+1}(\hat{\pi}, \hat{\pi}_0) = \min_{x_t, y_t, z_t, \theta_{t+1}} \quad & \hat{\pi}_0 f_t(x_t, y_t) + \hat{\pi}^\top z_t + \hat{\pi}_0 \theta_{t+1} \\
\text{s.t.} \quad & (x_{t}, y_{t}) \in \Phi_{n}(z_{t}, \xi_{t}) \\
& \theta_{t+1} \ge \psi_{t+1}^{i}(x_{t}) \\
& z_{t} \in Z_{t-1}, \quad x_{t} \in X_{t}
\end{align}
设求解上述内层模型获得的最优原问题决策解为 $(\tilde{x}_t, \tilde{y}_t, \tilde{z}_t, \tilde{\theta}_{t+1})$。

\paragraph{次梯度（Subgradient）计算：}
根据对偶函数的微分性质，$\omega_t^{i+1}$ 在当前乘子中心 $(\hat{\pi}, \hat{\pi}_0)$ 处的偏导数（次梯度）完全由原变量的最优解确定：
\begin{equation}
\frac{\partial \omega_t^{i+1}(\hat{\pi}, \hat{\pi}_0)}{\partial \pi} = \tilde{z}_t, \quad \frac{\partial \omega_t^{i+1}(\hat{\pi}, \hat{\pi}_0)}{\partial \pi_0} = f_t(\tilde{x}_t, \tilde{y}_t) + \tilde{\theta}_{t+1}
\end{equation}


\subsubsection{ 对偶值函数的线性割（Cut）更新表达式}
利用计算得到的次梯度，对偶值函数 $\omega_t^{i+1}(\pi, \pi_0)$ 的一阶泰勒支撑面（即对偶 Cut）表示为：
\begin{align}
\omega_t^{i+1}(\pi, \pi_0) &\le \omega_t^{i+1}(\hat{\pi}, \hat{\pi}_0) + \left(\frac{\partial \omega_t^{i+1}}{\partial \pi}\right)^\top (\pi - \hat{\pi}) + \frac{\partial \omega_t^{i+1}}{\partial \pi_0} (\pi_0 - \hat{\pi}_0) \nonumber \\
&= \omega_t^{i+1}(\hat{\pi}, \hat{\pi}_0) + \tilde{z}_t^\top (\pi - \hat{\pi}) + [f_t(\tilde{x}_t, \tilde{y}_t) + \tilde{\theta}_{t+1}] (\pi_0 - \hat{\pi}_0) \nonumber \\
&= \tilde{z}_t^\top \pi + [f_t(\tilde{x}_t, \tilde{y}_t) + \tilde{\theta}_{t+1}] \pi_0
\end{align}
注：最后一式利用了内层目标函数关于乘子的正齐次性结构（即 $\omega_t^{i+1}(\hat{\pi}, \hat{\pi}_0) = \hat{\pi}^\top \tilde{z}_t + \hat{\pi}_0 [f_t(\tilde{x}_t, \tilde{y}_t) + \tilde{\theta}_{t+1}]$）消除常数常项后展开得到。这与原文完全一致。

\subsubsection{外层对偶最大化问题的割平面标准型}
在外层优化中，我们将所有历史生成的对偶 Cut 作为一个分段线性的上包络函数加入系统中：
\begin{align}
\max_{\pi, \pi_0} \quad & \omega_t^{i+1}(\pi, \pi_0) - \pi^\top x_{t-1}^i - \pi_0 \theta_t^i \\
\text{s.t.} \quad & \pi_0 \ge 0 \\
& (\pi, \pi_0) \in \mathcal{L}_1 \text{ 或 } \mathcal{L}_\infty \text{ 或 } \text{LN} \quad (\text{有界乘子约束空间}) \\
& \omega_t^{i+1}(\pi, \pi_0) \le \tilde{z}_t^\top \pi + [f_t(\tilde{x}_t, \tilde{y}_t) + \tilde{\theta}_{t+1}] \pi_0 \quad (\text{历史对偶 Cut 集合})
\end{align}
求解该最大化问题后，所得的最优目标函数值被记为当前系统对偶阶段的**上界（UB）**。
相应地，对偶函数在历史评估点上的最大已知值构成系统的**下界（LB）**：

\begin{equation}
\mathrm{LB}^{k}
:=
\max_{j=1,\ldots,k}
\Bigl\{
\omega_t^{i+1}(\pi^{j},\pi_0^{j})
-
(\pi^{j})^\top x_{t-1}^{i}
-
\pi_0^{j}\theta_t^{i}
\Bigr\}.
\end{equation}

其中 $(\pi^{j},\pi_0^{j})$ 表示第 $j$ 次 Bundle 迭代中实际评估过的乘子点。

由于对偶函数值是真实计算得到的，因此 $\mathrm{LB}^{k}$ 始终是当前对偶最优值的有效下界，并且具有单调不下降性质。




\subsubsection{带有近端项的 Level（级别集）对偶步长更新}

Level 的妙处：用“目标值”代替“步长参数”Level 方法换了一种思路。它不直接在目标函数里加惩罚项去限制步长(调参困难)，而是直接在目标函数上画一条“合格线”（也就是 $\text{level}$）：


计算当前的对偶间隙 $\text{gap} = \text{UB} - \text{LB}$。利用缩放系数 $\text{level\_factor} \in (0,1)$，确定目标限制阈值（Level）：
\begin{equation}
\text{level} = \text{UB} - \text{gap} \times \text{level\_factor}
\end{equation}
为了维持对偶乘子的更新平滑度，在外层引入二次近端项（Proximal Term），求解如下形式的近端二次规划（QP）问题：
\begin{align}
\min_{\pi, \pi_0} \quad & \frac{1}{2} \|\pi - \hat{\pi}\|_2^2 + \frac{1}{2} (\pi_0 - \hat{\pi}_0)^2 \\
\text{s.t.} \quad & \omega_t^{i+1}(\pi, \pi_0) - \pi^\top x_{t-1}^i - \pi_0 \theta_t^i \ge \text{level} \\
& (\pi, \pi_0) \in \mathcal{L}_1 \text{ 或 } \mathcal{L}_\infty \text{ 或 } \text{LN} \\
& \omega_t^{i+1}(\pi, \pi_0) \le \omega^{\text{sub}} \quad (\text{利用历史 Cut 构成的对偶值约束})
\end{align}
求解该二次规划，即可生成下一轮更加稳定的新对偶乘子中心。




\section{正则化策略与范数边界约束}

\subsection{正则化策略}

在 SDDP 类算法的前向过程中，子问题的解（trial point）可能因割平面近似不精确而产生剧烈振荡，导致后向过程中生成的割质量下降、收敛缓慢。正则化策略通过在子问题目标函数中添加惩罚项 $\sigma_t \cdot \|z_t - x_{t-1}^i\|$，限制状态变量偏离前向传播中固定的状态值 $x_{t-1}^i$，从而稳定迭代过程。同时，正则化系数 $\sigma_t$ 还作为对偶乘子的范数边界 $B_t$，约束对偶乘子的绝对大小，防止对偶问题无界或数值不稳定。

\subsubsection{正则化项的数学形式}

在第 $t$ 阶段，给定固定的状态变量值 $x_{t-1}^i$（来自前向传播），正则化项为 $\sigma_t \cdot \|z_t - x_{t-1}^i\|$，其中 $\sigma_t$ 为正则化系数，范数 $\|\cdot\|$ 可选择 $\ell_1$ 或 $\ell_\infty$：

\paragraph{$\ell_1$ 范数：}
\begin{equation}
\|z_t - x_{t-1}^i\|_1 = \sum_{j=1}^{n_x} |z_{t,j} - x_{t-1,j}^i|
\end{equation}

$\ell_1$ 范数具有稀疏性诱导特性，倾向于使对偶乘子稀疏。

\paragraph{$\ell_\infty$ 范数：}
\begin{equation}
\|z_t - x_{t-1}^i\|_\infty = \max_{j=1,\dots,n_x} |z_{t,j} - x_{t-1,j}^i|
\end{equation}

$\ell_\infty$ 范数具有均匀性特性，约束所有状态变量分量的偏离程度一致。

\subsubsection{正则化在算法中的三个作用位置}

在 DynamicSDDiP.jl 中，正则化项出现在三个不同的位置，其数学形式和作用各不相同。

- 位置1：前向过程的原始子问题

在前向过程中，对 $t > 1$ 的阶段，求解子问题前先施加正则化。设 $x_{t-1}^i$ 为前向传播中固定的入状态值，$z_t$ 为 unfix 后的入状态变量（起拷贝变量的作用），正则化后的前向子问题为：

\begin{align}
\min_{x_t, y_t, \theta_{t+1}} \quad & f_{t}(x_{t}, y_{t}) + \theta_{t+1} + \sigma_t \cdot \|z_t - x_{t-1}^i\| \\
\text{s.t.} \quad & (x_{t}, y_{t}) \in \Phi_{n}(z_{t}, \xi_{t}) \\
& z_{t} = x_{t-1}^i \quad (\text{unfix 后松弛}) \\
& \theta_{t+1} \ge \psi_{t+1}^{i}(x_{t}) \\
& z_{t} \in X_{t-1}, \quad x_{t} \in X_{t}
\end{align}

\textbf{作用}：稳定前向过程的 trial point，避免因割平面近似不精确导致的状态变量跳跃。求解后，正则化项被移除，恢复原始子问题模型，继续沿场景路径前进。

- 位置2：后向过程的原始子问题（Primal Bound）

在后向过程中，当使用 `LagrangianDuality` 或 `UnifiedLagrangianDuality` 时，在进入 Lagrangian 对偶求解之前，先对原始子问题施加正则化并求解，获取带正则化的原始目标值 $\text{primal\_obj}$。其数学形式与位置1完全相同：

\begin{align}
\text{primal\_obj} = \min_{x_t, y_t, \theta_{t+1}} \quad & f_{t}(x_{t}, y_{t}) + \theta_{t+1} + \sigma_t \cdot \|z_t - x_{t-1}^i\| \\
\text{s.t.} \quad & \text{同位置1的约束}
\end{align}

\textbf{作用}：$\text{primal\_obj}$ 作为 Lagrangian 对偶外层问题的目标值上界（\texttt{obj\_bound}），用于收敛判定和数值稳定。求解后，正则化项被移除，恢复原始子问题，再进入 Lagrangian 对偶的迭代求解。

\textbf{注意}：在使用 `NormBound` 或 `BothBounds` 的对偶界方式时，$\text{primal\_obj}$ 中的正则化目标值提供对偶乘子的目标值约束；而在 `ValueBound` 方式下，$\text{primal\_obj}$ 直接作为 Lagrangian 对偶上界。

- 位置3：增广 Lagrangian 内层松弛（Augmented Lagrangian，仅经典框架，默认关闭）

当配置中显式启用 \texttt{augmented = true} 时，在经典 Lagrangian 对偶框架的内层松弛问题中加入正则化项。给定乘子 $\pi_k$，内层松弛问题修正为：

\begin{align}
L_k = \min_{x_t, y_t, z_t, \theta_{t+1}} \quad & f_t(x_t, y_t) + \theta_{t+1} - \pi_k^\top (z_t - x_{t-1}^i) + \rho \cdot \|z_t - x_{t-1}^i\| \\
\text{s.t.} \quad & (x_{t}, y_{t}) \in \Phi_{n}(z_{t}, \xi_{t}) \\
& \theta_{t+1} \ge \psi_{t+1}^{i}(x_{t}) \\
& z_{t} \in X_{t-1}, \quad x_{t} \in X_{t}
\end{align}

其中 $\rho = \sigma_t$ 为增广系数，$\|z_t - x_{t-1}^i\|$ 为所选范数（代码中只有$\ell_1$范数，即绝对值求和的形式 ）。

\textbf{作用}：增广 Lagrangian 方法通过在内层松弛中同时惩罚约束违反程度（$\rho \cdot \|z_t - x_{t-1}^i\|$），使对偶收敛更加稳定。

\textbf{重要限制}：此选项\textbf{仅在经典 Lagrangian 对偶框架}（\texttt{LagrangianDuality}）中可用，\textbf{不适用于统一框架}（\texttt{UnifiedLagrangianDuality}）。且默认关闭，需在 \texttt{CutGenerationRegime} 的 \texttt{duality\_regime} 中显式设置 \texttt{augmented = true} 方可启用。

\subsubsection{三个位置的对比总结}

\begin{table}[htbp]
\centering
\caption{正则化项在算法中的三个作用位置对比}
\label{tab:regularization_positions}
\begin{tabular}{|c|c|c|c|c|}
\hline
\textbf{位置} & \textbf{作用对象} & \textbf{数学形式} & \textbf{主要作用} & \textbf{是否默认启用} \\
\hline
位置1 & 前向 Primal & $f_t + \theta_{t+1} + \sigma_t \|z_t - x_{t-1}^i\|$ & 稳定 trial point & 是（若启用 Regularization） \\
\hline
位置2 & 后向 Primal & $f_t + \theta_{t+1} + \sigma_t \|z_t - x_{t-1}^i\|$ & 提供对偶上界 primal\_obj & 是（若启用 Regularization） \\
\hline
位置3 & 后向 Lagrangian 内层 & $(f_t + \theta_{t+1}) - \pi^\top h + \rho \|h\|$ & 增广收敛稳定性 & 否（需 augmented=true） \\
\hline
\end{tabular}
\end{table}

\subsubsection{正则化系数与对偶边界的关系}

正则化系数 $\sigma_t$ 不仅用于惩罚状态变量的偏离，同时也作为对偶乘子的范数边界（Dual Bound）。

根据拉格朗日对偶理论，正则化项 $\sigma_t \cdot \|z_t - x_{t-1}^i\|$ 的系数 $\sigma_t$ 对应于对偶乘子 $\pi$ 的最大可能值。因此，我们有：

\begin{equation}
B_t = \sigma_t
\end{equation}

其中 $B_t$ 为第 $t$ 阶段的对偶边界值。该边界值在范数边界约束中被直接使用（详见第\ref{sec:norm_bound}节）。

\paragraph{正则化策略配置：}

定义正则化策略配置 $\mathcal{R}_t = (\sigma_t, \|\cdot\|, \|\cdot\|_{\text{lifted}})$，其中：
\begin{itemize}
    \item $\sigma_t$：第 $t$ 阶段的正则化系数；
    \item $\|\cdot\|$：前向和后向 Primal 子问题中使用的正则化范数（L1 或 L∞）；
    \item $\|\cdot\|_{\text{lifted}}$：后向过程中范数边界约束使用的范数（L1 或 L∞），用于约束对偶乘子的绝对大小。
\end{itemize}

\paragraph{正则化系数的自适应调整：}

在迭代过程中，$\sigma_t$ 可根据算法收敛情况进行调整。定义缩放因子 $\sigma_{\text{factor}} > 0$，当 Sigma 测试判定正则化不充分时，通过下式增大 $\sigma_t$：

\begin{equation}
\sigma_t^{\text{new}} = \sigma_{\text{factor}} \cdot \sigma_t
\end{equation}



\subsection{范数边界}
\subsubsection{范数边界的基本形式}
范数边界约束（Norm Bound Constraints）直接限制对偶乘子 $(\pi, \pi_0)$ 的大小，确保对偶问题有唯一解且数值稳定。以下描述均针对统一 Lagrangian 对偶框架。
(经典lagrangian没有 pi0，因此形式会有差异)

范数边界约束的核心思想是限制 $\pi$ 相对于 $\pi_0$ 的比值：

\begin{equation}
\frac{\|\pi\|}{\pi_0} \le B_t
\end{equation}

其中 $B_t = \sigma_t$ 为对偶边界值，由正则化系数决定。

\subsubsection{范数边界约束的两种形式}
$\|\cdot\|$ 的具体范数类型由正则化所用的范数决定，存在对偶对应关系：

\begin{itemize}
\item 正则化使用 $\ell_1$ 范数 $\Rightarrow$ 范数边界限制 $\ell_\infty$ 范数（逐元素上界）：
\begin{equation}
\|\pi\|_\infty \le B_t \cdot \pi_0
\end{equation}
等价于逐元素约束：
\begin{equation}
|\pi_j| \le B_t \cdot w_j \cdot \pi_0, \quad \forall j = 1,\dots,n_x
\end{equation}
其中 $w_j$ 为第 $j$ 个状态变量对应的权重系数（在二值近似下 $w_j = 2^{k-1}\beta$，在无状态近似下 $w_j = 1$）。

\item 正则化使用 $\ell_\infty$ 范数 $\Rightarrow$ 范数边界限制 $\ell_1$ 范数（加权求和上界）：
\begin{equation}
\|\pi\|_{w,1} \le B_t \cdot \pi_0
\end{equation}
即：
\begin{equation}
\sum_{j=1}^{n_x} w_j \cdot |\pi_j| \le B_t \cdot \pi_0
\end{equation}
\end{itemize}

\textbf{无正则化时，$B_t = \infty$，上述约束自动失效，对偶有界性由归一化约束单独保证。}




\subsubsection{约束的几何解释}

范数边界约束在 $(\pi_0, \pi)$ 空间中定义了一个有界区域：

\begin{itemize}
    \item $\ell_1$ 时定义了一个\textbf{棱锥}：所有对偶变量必须满足 $|\pi_j| \le B_t \cdot w_j \cdot \pi_0$；
    \item $\ell_\infty$ 时定义了一个\textbf{多面体锥}：所有对偶变量必须满足 $\sum_{j=1}^{n_x} w_j \cdot |\pi_j| \le B_t \cdot \pi_0$。
\end{itemize}

\subsubsection{权重系数$w_j$的计算}

权重系数 $w_j$ 用于缩放不同状态变量的对偶乘子边界，反映状态变量的相对重要性。

当状态变量采用二进制近似（Binary Approximation）时，权重系数由下式计算：

\begin{equation}
w_j = 2^{k_j - 1} \cdot \beta_j
\end{equation}

其中：
\begin{itemize}
    \item $k_j$：状态变量 $x_j$ 的二进制分解位数；
    \item $\beta_j$：状态变量 $x_j$ 的二进制精度（每个二进制位代表的数值）。
\end{itemize}

在二进制近似中，原始整数状态 $x_j$ 被分解为：

\begin{equation}
x_j = \sum_{m=1}^{k_j} 2^{m-1} \cdot \beta_j \cdot b_{j,m}
\end{equation}

其中 $b_{j,m}$ 为二进制变量（0 或 1）。权重系数 $w_j$ 反映了二进制位对原始状态的贡献程度：
\begin{itemize}
    \item 高位（$m$ 大）的权重更大，因为 $2^{m-1}$ 更大；
    \item 精度 $\beta_j$ 越高，权重越大。
\end{itemize}

\textbf{当不使用二进制近似（No State Approximation）时，权重系数默认为 1：}

\begin{equation}
w_j = 1, \quad \forall j = 1,\dots,n_x
\end{equation}

\section{对偶乘子的 归一化 约束}

在统一 Lagrangian 对偶框架的 Outer Problem 中，规范化约束（Normalization Constraint）和范数边界约束（Norm Bound Constraint）同时生效，共同定义对偶乘子的可行域。\textbf{注意}：规范化约束仅存在于统一 Lagrangian 对偶框架（\texttt{UnifiedLagrangianDuality}）中，经典 Lagrangian 对偶框架不包含规范化约束。

\subsection{norm 约束形式}

规范化约束的作用是对对偶乘子 $(\pi_0, \pi)$ 进行归一化，确保外层问题有界，同时使生成的割为"最深割"（Deep Cut）。不同归一化方式对应不同的规范化约束：

\begin{itemize}
    \item $\mathcal{L}_1$ 归一化（\texttt{L1\_Deep}）：
    \begin{equation}
    \pi_0 + \sum_{j=1}^{n_x} |\pi_j| \le 1
    \end{equation}

    \item $\mathcal{L}_\infty$ 归一化（\texttt{Linf\_Deep}）：
    \begin{equation}
    \pi_0 \le 1, \quad |\pi_j| \le 1, \quad \forall j
    \end{equation}

    \item $\mathcal{L}_2$ 归一化（\texttt{L2\_Deep}）：
    \begin{equation}
    \pi_0^2 + \sum_{j=1}^{n_x} \pi_j^2 \le 1
    \end{equation}

    \item Core 类归一化（反极割）：
    \begin{equation}
    \omega_0 \pi_0 + \sum_{j=1}^{n_x} \omega_j \pi_j \le 1
    \end{equation}
    其中 $(\omega_0, \omega)$ 为核心点方向系数。
\end{itemize}

\paragraph{代码实现中的松弛：}

在代码中，对偶乘子被拆分为 $\pi = \pi^+ - \pi^-$（$\pi^+, \pi^- \ge 0$）。以 $\mathcal{L}_1$ 归一化为例，代码实现为：
\begin{equation}
\pi_0 + \sum_{j=1}^{n_x} (\pi_j^+ + \pi_j^-) \le 1
\end{equation}
由于 $|\pi_j| = |\pi_j^+ - \pi_j^-| \le \pi_j^+ + \pi_j^-$，代码约束是数学约束 $\pi_0 + \sum |\pi_j| \le 1$ 的\textbf{充分条件}（更松弛），而非等价形式。当 $\pi_j^+$ 和 $\pi_j^-$ 不同时为正时（这在最优解中成立），两者等价。

\subsection{完整的约束体系}

以选择 $\mathcal{L}_1$ 归一化和 \texttt{norm\_lifted} = $\ell_1$（限制 $\ell_\infty$ 范数）为例，统一框架外层问题的完整约束体系为：

\begin{align}
\max_{\pi, \pi_0} \quad & \omega_t^{i+1}(\pi, \pi_0) - \pi^\top x_{t-1}^i - \pi_0 \theta_t^i \\
\text{s.t.} \quad & \pi_0 \ge 0 \\
& \pi_0 + \sum_{j=1}^{n_x} |\pi_j| \le 1 \quad (\mathcal{L}_1 \text{ 规范化约束}) \\
& |\pi_j| \le B_t \cdot w_j \cdot \pi_0, \quad \forall j \quad (\text{范数边界约束}) \\
& \omega_t^{i+1}(\pi, \pi_0) \le \tilde{z}_t^\top \pi + [f_t(\tilde{x}_t, \tilde{y}_t) + \tilde{\theta}_{t+1}] \pi_0 \quad (\text{历史对偶 Cut})
\end{align}

\subsection{约束的作用分工}

\begin{itemize}
    \item \textbf{规范化约束}：定义对偶变量的相对比例（单位球），确保生成"最深"的割，保证外层问题有界；
    \item \textbf{范数边界约束}：限制对偶变量的绝对大小，利用正则化系数 $B_t = \sigma_t$ 防止数值发散；
    \item \textbf{两者协同}：规范化约束定义可行域的形状，范数边界约束定义可行域的大小，共同确保对偶问题有唯一解且数值稳定。
\end{itemize}

\subsection{可行域的几何描述}

最终可行域是规范化约束的单位球与范数边界约束的锥的交集：

\begin{equation}
\mathcal{F} = \{ (\pi, \pi_0) \mid \|\tilde{\pi}\| \le 1 \} \cap \{ (\pi, \pi_0) \mid \frac{\|\pi\|}{\pi_0} \le B_t \cdot w \}
\end{equation}

其中 $\tilde{\pi} = (\pi_0, \pi_1, \dots, \pi_{n_x})$。

\section{算法流程}

将正则化和范数边界约束集成到 Level Bundle 算法中的完整流程如下：

\begin{enumerate}
    \item \textbf{初始化}：设置正则化系数 $\sigma_t$、权重系数 $w_j$、对偶边界 $B_t = \sigma_t$；
    \item \textbf{内层求解}：求解带正则化的拉格朗日松弛子问题，获得次梯度；
    \item \textbf{添加切平面}：将次梯度转化为对偶 Cut，添加到 Outer Problem；
    \item \textbf{添加范数边界约束}：根据选择的范数类型，添加相应的边界约束；
    \item \textbf{求解 Outer Problem}：求解带所有约束的对偶最大化问题，获得 UB；
    \item \textbf{计算 LB}：在当前乘子点评估实际松弛收益，获得 LB；
    \item \textbf{收敛判定}：检查 $|\text{UB} - \text{LB}| \le \text{gap\_tol} \cdot |\text{UB}|$；
    \item \textbf{Level 更新}：计算 level，求解带近端项的二次规划，获得新的乘子中心；
    \item \textbf{迭代}：回到步骤 2，直到收敛或达到迭代上限。
\end{enumerate}

\section{ 参数选择建议}

\subsection{8.1 正则化系数 $\sigma_t$ 的选择}

$\sigma_t$ 的选择需要平衡收敛速度和数值稳定性：

\begin{itemize}
    \item $\sigma_t$ 过小：正则化效果不明显，可能导致对偶乘子发散；
    \item $\sigma_t$ 过大：惩罚过度，可能限制搜索范围，导致收敛到次优解。
\end{itemize}

\textbf{推荐策略}：从较小的值开始（如 $\sigma_t = 10$），根据算法收敛情况逐步调整。

\subsection{ 范数类型的选择}

\begin{itemize}
    \item \textbf{前向传播范数}：建议选择 $\ell_1$ 范数，具有稀疏性诱导特性；
    \item \textbf{后向传播范数}（范数边界约束）：
        \begin{itemize}
            \item $\ell_1$ 范数边界：限制 $\ell_\infty$ 范数，适用于一般情况；
            \item $\ell_\infty$ 范数边界：限制加权 $\ell_1$ 范数，适用于数值稳定性要求高的情况。
        \end{itemize}
\end{itemize}

\subsection{ 权重系数的选择}

\begin{itemize}
    \item 二进制近似：使用公式 $w_j = 2^{k_j - 1} \cdot \beta_j$；
    \item 非二进制近似：使用 $w_j = 1$；
    \item 自定义权重：根据状态变量的实际物理意义设置权重。
\end{itemize}

\section{ 总结}

正则化策略和范数边界约束是 Level Bundle 算法中不可或缺的组成部分：

\begin{itemize}
    \item \textbf{正则化策略}通过在目标函数中添加惩罚项，限制状态变量的偏离，同时提供对偶乘子的理论上界；
    \item \textbf{范数边界约束}直接限制对偶乘子的大小，确保对偶问题有唯一解且数值稳定；
    \item \textbf{权重系数}用于缩放不同状态变量的边界，反映状态变量的相对重要性；
    \item \textbf{规范化约束与范数边界约束协同作用}，共同定义对偶乘子的可行域。
\end{itemize}

合理选择正则化系数、范数类型和权重系数，可以显著提高算法的数值稳定性和收敛速度。






\appendix

\section{附录：正则化解决对偶无界的数学原理}

本附录详细解释为什么引入正则化项能够解决对偶问题无界以及对偶乘子发散的问题。

\subsection{A.1 对偶问题无界的数学原因}

首先回顾原始的拉格朗日对偶问题：

\begin{equation}
\omega_t^{i+1}(\pi, \pi_0) = \min_{x_t, y_t, z_t, \theta_{t+1}} \quad \pi_0 f_t(x_t, y_t) + \pi^\top z_t + \pi_0 \theta_{t+1}
\end{equation}

当外部进行最大化时：

\begin{equation}
\max_{\pi, \pi_0 \ge 0} \quad \omega_t^{i+1}(\pi, \pi_0) - \pi^\top x_{t-1}^i - \pi_0 \theta_t^i
\end{equation}

\textbf{问题所在}：对于每个固定的状态解 $(x_t, y_t, z_t, \theta_{t+1})$，上述松弛目标是关于 $\pi$ 和 $\pi_0$ 的\textbf{线性函数}。

\textbf{具体例子}：
\begin{itemize}
    \item 设某个内点原问题可行解满足 $\tilde{z}_t \neq 0$
    \item 则此时对偶函数形式为 $\omega_t^{i+1}(\pi, \pi_0) = \pi^\top \tilde{z}_t + \text{常数}$
    \item 外部最大化目标函数随之变为：$\pi^\top \tilde{z}_t - \pi^\top x_{t-1}^i = \pi^\top (\tilde{z}_t - x_{t-1}^i) + \text{常数}$
    \item 如果向量 $(\tilde{z}_t - x_{t-1}^i)$ 的某个分量为正，则对偶优化为了求最大值，倾向于让 $\pi$ 沿该方向趋向于 $+\infty$，导致目标函数无上界（无界）。
\end{itemize}

\subsection{A.2 正则化如何约束对偶乘子}

\subsubsection{A.2.1 正则化改变了内层问题的结构}

加入正则化后：

\begin{equation}
\omega_t^{i+1}(\pi, \pi_0) = \min_{x_t, y_t, z_t, \theta_{t+1}} \quad \pi_0 f_t(x_t, y_t) + \pi^\top z_t + \pi_0 \theta_{t+1} + \sigma_t \|z_t - x_{t-1}^i\|
\end{equation}

\textbf{关键变化}：内层问题不再是简单的线性函数最小化，而是一个\textbf{严格凸函数}（线性函数 + 凸范数项）的最小化。

\subsubsection{A.2.2 凸优化问题的对偶性质}

考虑简化的正则化子问题（暂时忽略 $f_t$ 和 $\theta_{t+1}$ 的常数影响）：

\begin{equation}
\min_{z_t \in Z_{t-1}} \quad \pi^\top z_t + \sigma \|z_t - x_{t-1}^i\| 
\end{equation}

\textbf{引理}：对于凸优化问题，其对偶函数 $\omega(\pi)$ 必然是一个\textbf{凹函数}，且满足一阶支撑不等式：

\begin{equation}
\omega(\pi) \le \omega(\pi^*) + g^\top (\pi - \pi^*), \quad \forall g \in \partial \omega(\pi^*)
\end{equation}

更重要的是，\textbf{由于原问题的可行域有界且引入了范数惩罚，其对偶凹函数的次梯度是有界的}。

\subsubsection{A.2.3 次梯度有界的数学证明}

以 $\ell_1$ 正则化的情况为例，考虑关于 $z$ 的优化目标：

\begin{equation}
\phi(z) = \pi^\top z + \sigma \|z - x\|_1
\end{equation}

\textbf{最优性条件的次梯度集合}：

\begin{equation}
0 \in \pi + \sigma \cdot \partial \|z^* - x\|_1
\end{equation}

其中 $\ell_1$ 范数的次梯度集合 $\partial \|z - x\|_1$ 的每个分量为：

\begin{equation}
\left[\partial \|z - x\|_1\right]_j = \begin{cases} \text{sign}(z_j - x_j) & \text{if } z_j \neq x_j \\ [-1, 1] & \text{if } z_j = x_j \end{cases}
\end{equation}

\textbf{关键结论}：由于范数次梯度的无穷范数上限恒为 $1$（即 $\|\partial \|z - x\|_1\|_\infty \le 1$），这意味着最优点处的对偶变量满足：

\begin{equation}
\|\pi\|_\infty \le \sigma
\end{equation}

通过这种方式，乘子 $\pi$ 的范围被惩罚系数 $\sigma$ 牢牢约束住了。

\subsection{A.3 从 Legendre-Fenchel 共轭角度理解}

\subsubsection{A.3.1 共轭函数的性质}

任意范数 $\|z\|$ 的 Fenchel 共轭函数定义为：

\begin{equation}
\|u\|_* = \sup_{z} \{ u^\top z - \|z\| \}
\end{equation}

对于 $\ell_1$ 范数，其共轭为对应对偶范数（即 $\ell_\infty$ 范数）的指示函数：

\begin{equation}
\|u\|_* = \begin{cases} 0 & \text{if } \|u\|_\infty \le 1 \\ +\infty & \text{otherwise} \end{cases}
\end{equation}

\subsubsection{A.3.2 正则化项的等价形式}

利用双重对偶性，正则化项可以改写为极大化形式：

\begin{equation}
\sigma \|z - x\| = \sup_{u: \|u\|_* \le \sigma} u^\top (z - x)
\end{equation}

这意味着，原小化对偶函数可转化为：

\begin{equation}
\omega(\pi) = \min_z \sup_{u: \|u\|_* \le \sigma} (\pi + u)^\top z - u^\top x
\end{equation}

\textbf{数学含义}：在原目标中引入近端正则项，等价于在对偶最大化求解时，在对偶变量 $\pi$ 上施加了一个\textbf{有界的对偶扰动} $u$，而 $u$ 的边界半径正好由 $\sigma$ 决定。

\subsubsection{A.3.3 对偶函数的上界}

因此，正则化后的对偶函数有界的性质由以下关系式托底：

\begin{equation}
\omega(\pi, \pi_0) \le \min_{x \in \text{dom}(f)} \left[ \pi_0 f(x) + \sigma \|x - x^i\| \right] + \text{常数}
\end{equation}

由于 $\sigma \|x - x^i\| \ge 0$，该对偶函数从上方被一个\textbf{有限上界值}所约束。

\subsection{A.4 几何直观理解}

\subsubsection{A.4.1 无正则化时的可行域}

\begin{figure}[htbp]
\centering
\begin{tikzpicture}
\draw[->] (0,0) -- (5,0) node[right] {$\pi$};
\draw[->] (0,0) -- (0,5) node[above] {$\pi_0$};
\draw[dashed] (2,0) -- (2,4);
\draw[dashed] (0,2) -- (4,2);
\draw[red, thick] (1,4) -- (4,1);
\node at (3.2,3.2) {无界区域};
\node at (3,2.7) {$\uparrow$};
\node at (3.5,2.2) {对偶函数值可以};
\node at (3.5,1.8) {沿此方向无限增大};
\end{tikzpicture}
\caption{无正则化时的对偶开放可行域（无上界）}
\end{figure}

如图所示，此时对偶乘子 $\pi$ 可以沿某个线性切平面方向无限增大而不触发边界阻碍。

\subsubsection{A.4.2 加入正则化后的可行域}

\begin{figure}[htbp]
\centering
\begin{tikzpicture}
\draw[->] (0,0) -- (5,0) node[right] {$\pi$};
\draw[->] (0,0) -- (0,5) node[above] {$\pi_0$};
\draw[thick, blue] (0,4) -- (4,0);
\draw[thick, blue] (0,0) -- (4,0) -- (0,4) -- cycle;
\filldraw[black] (1.5,1.5) circle (2pt);
\node at (1.5,1.5) [above right] {$\pi^*$};
\node at (2.2,0.8) {有界级别集域};
\draw[<->] (0, -0.3) -- (4, -0.3) node[midway, below] {边界半径 $\propto \sigma$};
\end{tikzpicture}
\caption{加入正则化项后收缩形成的闭合有界对偶区域}
\end{figure}

正则化项配合 Level-Set（级别集）将解空间从无界区域强制收紧到了一个\textbf{闭合有界凸集}内，确保了对偶极大化问题必然存在唯一的有限最优解。

\subsection{A.5 为什么正则化系数 $\sigma$ 就是对偶边界 $B$}

\subsubsection{A.5.1 从 KKT 最优性条件理解}

对于基础正则化问题：
\begin{equation}
\min_z \quad \pi^\top z + \sigma \|z - x\|
\end{equation}
其一阶 KKT 导数零点条件要求存在次梯度特征向量 $g \in \partial \|z^* - x\|$ 使得：
\begin{equation}
\pi + \sigma g = 0
\end{equation}
由于单位范数次梯度的对偶范数性质满足 $\|g\|_* \le 1$，所以两边取对偶范数直接得到：
\begin{equation}
\|\pi\|_* \le \sigma
\end{equation}
特别地，当原空间采用 $\ell_1$ 范数正则化时，其对偶范数即为无穷范数 $\|\cdot\|_\infty$，故得到：
\begin{equation}
\|\pi\|_\infty \le \sigma
\end{equation}
这在数学上严谨地证明了：**近端惩罚系数 $\sigma$ 构成了对偶乘子空间的硬性边界。**

\subsubsection{A.5.2 推广到完整的多阶段问题}

对于带有阶段状态约束的完整多阶段拉格朗日松弛问题：
\begin{equation}
\omega(\pi, \pi_0) = \min_{x,y,z,\theta} \left\{ \pi_0 f(x,y) + \pi^\top z + \pi_0 \theta + \sigma \|z - x^i\| \right\}
\end{equation}
应用 KKT 可得状态乘子与标量乘子之间的显式界限关系：
\begin{equation}
\|\pi\|_\infty \le \sigma \cdot \pi_0
\end{equation}
也就是说，各维度的对偶乘子分量被完全限制在以标量乘子和正则化系数缩放的域内：
\begin{equation}
|\pi_j| \le \sigma \cdot \pi_0, \quad \forall j
\end{equation}

\subsection{A.6 总结：正则化解决无界问题的完整链条}

正则化通过以下严格的因果逻辑链条解决对偶无界和收敛发散问题：
\begin{enumerate}
    \item \textbf{原始问题无界}：在没有加入近端正则化前，对偶极大化 $\max \omega(\pi)$ 的内层是一个纯线性的最小化问题，面对割平面的不完美逼近极易在某些无界射线方向上发散。
    \item \textbf{引入近端正则项}：通过改写为 $\omega(\pi) = \min (\pi^\top z + \sigma\|z-x\| + \dots)$，人为改变了内层极小化规划的曲率。
    \item \textbf{凸优化性质引入}：内层问题由线性规划转变为凸优化规划，使得生成的对偶函数具有天然的凹性，并且其外包络次梯度变得整体可控。
    \item \textbf{次梯度边界锁定}：利用一阶最优性条件（KKT），将乘子强行和范数次梯度的凸集边界挂钩，导出 $\|\pi\|_* \le \sigma$。
    \item \textbf{对偶有界性获证}：由于对偶乘子被系数 $\sigma$ 限制在有界区域内，算法避免了数值振荡，确保了 SDDiP 反向传播过程中 Cut 构造的稳健性。
\end{enumerate}

\begin{figure}[htbp]
\centering
\begin{tikzpicture}
% 重新修正了带有 TikZ 相对布局的节点定位，消除了 node distance 的编译冲突
\node (start) [draw, rectangle, align=center] {1. 原始问题无界 \\ $\max \omega(\pi)$ 呈纯线性极值跳跃};
\node (reg) [draw, rectangle, align=center, below=0.6cm of start] {2. 加入正则化项 \\ $\omega(\pi) = \min(\pi^\top z + \sigma\|z-x\|)$};
\node (convex) [draw, rectangle, align=center, below=0.6cm of reg] {3. 凸优化性质引入 \\ 函数转变为严格凸极小化问题};
\node (subgrad) [draw, rectangle, align=center, below=0.6cm of convex] {4. 次梯度边界锁定 \\ 触发最优性导数零点 $\pi + \sigma g = 0$};
\node (bound) [draw, rectangle, align=center, below=0.6cm of subgrad] {5. 对偶边界约束确立 \\ $\|\pi\|_\infty \le \sigma \cdot \pi_0$ 确保解有界};

\draw[->, thick] (start) -- (reg);
\draw[->, thick] (reg) -- (convex);
\draw[->, thick] (convex) -- (subgrad);
\draw[->, thick] (subgrad) -- (bound);
\end{tikzpicture}
\caption{正则化项驱动对偶乘子有界收敛的完整因果逻辑链}
\end{figure}

\subsection{A.7 形式化表达总结}

\begin{table}[htbp]
\centering
\caption{正则化机制的数学表达与核心作用对比}
\begin{tabular}{|c|c|c|}
\hline
\textbf{算法机制} & \textbf{标准数学表达} & \textbf{对系统收敛的作用} \\
\hline
原始未松弛对偶 & $\max \omega(\pi) = \max \min (\pi^\top z + \dots)$ & 线性极值搜索，乘子极易跳跃发散 \\
\hline
近端正则化 & $\omega(\pi) = \min (\pi^\top z + \sigma\|z-x\| + \dots)$ & 重构为凸极小问题，引入空间曲率 \\
\hline
凸包络次梯度 & $\nabla \omega(z^*) \in \pi + \sigma \cdot \partial \|z^* - x\|$ & 限制了割平面支撑切线的最大摆动斜率 \\
\hline
KKT 稳定条件 & $\pi + \sigma g = 0, \ \|g\|_* \le 1$ & 将乘子边界与正则化步长直接挂钩 \\
\hline
范数边界限制 & $\|\pi\|_\infty \le \sigma \cdot \pi_0$ & 锁定了对偶级集范围，消除了无界解 \\
\hline
\end{tabular}
\end{table}

\textbf{核心结论}：**近端正则化系数 $\sigma$ 实际上就是控制对偶空间防发散边界 $B$ 的数学本质。** 它通过将原始脆弱的线性极大极小跳跃重构为平滑的凸极小-凹极大（Saddle-Point）问题，借助凸优化次梯度有界的优良理论基石，从根本上杜绝了多阶段随机规划中对偶乘子的剧烈振荡，保障了收敛效率。

\section{附录：正则化系数 $\sigma$ 的选择逻辑}

本附录详细描述 DynamicSDDiP.jl 项目中正则化系数 $\sigma$ 的选择逻辑和自适应调整机制。

\subsection{B.1 Sigma 的初始化逻辑}

在算法初始化阶段（对应 `algorithmMain.jl`），$\sigma$ 的初始值按照以下规则设置：

\begin{equation}
\sigma_t = \begin{cases} 
0 & \text{当 } t = 1 \\
1.0 & \text{当 } t > 1 
\end{cases}
\end{equation}

\textbf{设计理由}：
\begin{itemize}
    \item 第一阶段（$t=1$）不需要正则化，因为没有前一阶段的状态变量需要解耦与引入对偶约束；
    \item 其他阶段（$t>1$）初始化为 $1.0$，这是一个保守的默认值，后续通过 Sigma 测试（Sigma Test）实施自适应调整。
\end{itemize}

对应的核心代码实现：
\begin{verbatim}
if isa(regime, DynamicSDDiP.Regularization) && isempty(regime.sigma)
    for (node_index, _) in model.nodes
        if node_index == 1
            # first stage requires no regularization
            push!(regime.sigma, 0.0)
        else
            push!(regime.sigma, 1.0)
        end
    end
end
\end{verbatim}

\subsection{B.2 Sigma 的自适应调整机制（Sigma Test）}

在每次外层迭代达到收敛条件后，算法会执行 Sigma 测试（`sigmaTest.jl`），检查当前 $\sigma$ 是否足够大。

\subsubsection{B.2.1 Sigma 测试流程}

Sigma 测试的核心思想是比较正则化子问题和非正则化子问题的目标函数解：

\begin{enumerate}
    \item \textbf{求解正则化问题}：
        \begin{equation}
        \min_{x_t, y_t, z_t} \quad f_t(x_t, y_t) + \sigma_t \|z_t - x_{t-1}^i\|
        \end{equation}
        得到最优目标函数值 $\text{obj}_{\text{reg}}$；
    
    \item \textbf{求解非正则化问题}：
        \begin{equation}
        \min_{x_t, y_t, z_t} \quad f_t(x_t, y_t)
        \end{equation}
        得到目标函数值 $\text{obj}_{\text{non-reg}}$；
    
    \item \textbf{比较两个目标值}：
        \begin{equation}
        \text{if } |\text{obj}_{\text{reg}} - \text{obj}_{\text{non-reg}}| > \epsilon \quad \Rightarrow \quad \text{判定不一致，增大 } \sigma
        \end{equation}
\end{enumerate}

\subsubsection{B.2.2 Sigma 调整公式}

当 Sigma 测试失败（即正则化问题和非正则化问题的解不一致，说明当前的范数惩罚力度不足以束缚变量的漂移）时，$\sigma$ 按照以下公式进行放大调整：

\begin{equation}
\sigma_t^{\text{new}} = \sigma_t^{\text{old}} \times \sigma_{\text{factor}}
\end{equation}

其中 $\sigma_{\text{factor}}$ 在系统中默认取值为 $5.0$。

对应的核心调整代码：
\begin{verbatim}
# sigmaTest.jl
if !isapprox(reg_results.objective, non_reg_results.objective)
    sigma = algo_params.regularization_regime.sigma[node_index]
    sigma_factor = algo_params.regularization_regime.sigma_factor
    sigma = sigma * sigma_factor
    sigma_increased = true
end

# algorithmMain.jl
if sigma_increased
    regularization_regime.sigma = regularization_regime.sigma * regularization_regime.sigma_factor
    sigma_increased = true
end
\end{verbatim}

\subsection{B.3 Sigma 与对偶边界的关系}

在 DynamicSDDiP.jl 系统构架中，正则化系数 $\sigma$ 会直接作为对偶乘子的无穷范数边界约束值（\texttt{lagrange\_preparation.jl}）：

\begin{equation}
B_t = \sigma_t
\end{equation}

\subsubsection{B.3.1 对偶边界的计算规则}

根据系统对偶边界模式 \texttt{dual\_bound\_regime} 的不同选择，边界的计算方式及限制表现为表 B.1 所示。

\begin{table}[htbp]
\centering
\caption{Dual Bound Regime 与对偶边界的关系模式对照表}
\label{tab:dual_bound_regime}
\begin{tabular}{|c|c|p{7.5cm}|}
\hline
\textbf{Dual Bound Regime} & \textbf{\texttt{dual\_bound} 限制值} & \textbf{边界作用说明} \\
\hline
\texttt{ValueBound} & $\infty$ & 只使用目标值范围截断，无乘子范数硬性约束 \\
\hline
\texttt{NormBound} & $\sigma_t$ & 只开启对偶乘子范数边界，$\|\pi\|_\infty \le \sigma_t \cdot \pi_0$ \\
\hline
\texttt{BothBounds} & $\sigma_t$ & 综合重叠使用目标值边界与乘子范数边界 \\
\hline
\texttt{NoRegularization} & $\infty$ & 无正则化状态下，对偶边界设为无穷大 \\
\hline
\end{tabular}
\end{table}

对应的边界获取核心代码实现如下：
\begin{verbatim}
function get_dual_bounds(node_index, algo_params, 
                         dual_bound_regime::DynamicSDDiP.NormBound)
    dual_bound = Inf
    if isa(algo_params.regularization_regime, DynamicSDDiP.NoRegularization)
        dual_bound = Inf
    else
        dual_bound = algo_params.regularization_regime.sigma[node_index]
    end
    return (obj_bound = Inf, dual_bound = dual_bound)
end
\end{verbatim}

\subsection{B.4 Sigma 选择的完整流程图}

\begin{figure}[htbp]
\centering
\begin{tikzpicture}[node distance=0.7cm and 1.2cm]
% 统一使用标准 TikZ 语法与显式命名控制
\node (start) [draw, rectangle, align=center] {算法开始};
\node (init) [draw, rectangle, align=center, below=of start] {初始化 $\sigma$ \\ $\sigma_1=0, \ \sigma_{t>1}=1$};
\node (iter) [draw, rectangle, align=center, below=of init] {外层收敛迭代寻找平衡点};
\node (conv) [draw, diamond, align=center, aspect=2, below=of iter] {收敛条件满足？};
\node (sigma_test) [draw, rectangle, align=center, below=of conv] {执行自适应 Sigma 测试};
\node (compare) [draw, diamond, align=center, aspect=2, below=of sigma_test] {正则目标 $\approx$ 非正则目标？};
\node (increase) [draw, rectangle, align=center, below=of compare] {$\sigma = \sigma \times 5$ \\ 触发连锁重置机制};
\node (update) [draw, rectangle, align=center, right=of compare] {更新当前 $\sigma$ \\ 为硬性对偶边界 $B_t$};
\node (end_node) [draw, rectangle, align=center, below=of update] {当前阶段算法终止};

% 连线控制
\draw[->, thick] (start) -- (init);
\draw[->, thick] (init) -- (iter);
\draw[->, thick] (iter) -- (conv);
\draw[->, thick] (conv) -- node[left] {是} (sigma_test);
\draw[->, thick] (conv.east) -- node[above] {否} ++(1.5,0) |- (iter.east);
\draw[->, thick] (sigma_test) -- (compare);
\draw[->, thick] (compare) -- node[left] {否} (increase);
\draw[->, thick] (compare) -- node[above] {是} (update);
\draw[->, thick] (increase.west) -- node[below] {重置迭代} ++(-1.2,0) |- (iter.west);
\draw[->, thick] (update) -- (end_node);
\end{tikzpicture}
\caption{DynamicSDDiP.jl 系统中 Sigma 自适应选择与对偶更新的完整逻辑因果链}
\label{fig:sigma_flowchart}
\end{figure}

\subsection{B.5 Sigma 调整触发的连锁重置反应}

由于改变 $\sigma$ 意味着原系统模型的拉格朗日凸包络曲率发生了改变，一旦 Sigma 测试判定需要调整，系统会强制触发以下连锁重置，以保障求解的稳健性：

\begin{enumerate}
    \item \textbf{丢弃旧解支撑点}：重置历史解状态，即设置 \texttt{previous\_solution = nothing} 和 \texttt{previous\_bound = nothing}；
    \item \textbf{重置边界精细化检查}：将边界判定标识重设为 \texttt{bound\_check = false}；
    \item \textbf{回滚内层收敛循环}：由于底层切平面基准改变，宣布当前收敛结果失效，强制重置 \texttt{result.has\_converged = false}，从而重新激发迭代；
    \item \textbf{更新对偶有界限制}：将新的 $\sigma$ 值作为下一轮对偶外层最大化时最新的乘子无穷范数边界。
\end{enumerate}

重置触发的核心控制代码片段：
\begin{verbatim}
if sigma_increased
    # reset previous values, as model is changed and convergence not achieved
    previous_solution = nothing
    previous_bound = nothing
    # binary refinement only when no sigma refinement has been made
    bound_check = false
    # no convergence, restart loop
    result.has_converged = false
end
\end{verbatim}

\subsection{B.6 参数配置与经验选择建议}

\subsubsection{B.6.1 初始 Sigma 值的设置}

\begin{itemize}
    \item \textbf{默认保守型（通用）}：$\sigma = [0, 1, 1, \dots, 1]$（第一阶段不予惩罚，其余阶段初始设为 $1$）；
    \item \textbf{小规模良态问题}：$\sigma = [0, 1, 1, \dots, 1]$；
    \item \textbf{中等规模耦合问题}：$\sigma = [0, 10, 10, \dots, 10]$；
    \item \textbf{超大规模/病态条件问题}：$\sigma = [0, 100, 100, \dots, 100]$。
\end{itemize}

\subsubsection{B.6.2 缩放因子 Sigma Factor 的选择}

\begin{itemize}
    \item \textbf{系统默认值}：$\sigma_{\text{factor}} = 5.0$；
    \item \textbf{激进跨步策略}：$\sigma_{\text{factor}} = 10.0$（适合期望快速试探出乘子有界可行域上限的情况）；
    \item \textbf{精细平滑策略}：$\sigma_{\text{factor}} = 2.0$（适合需要微调步长、避免过度惩罚导致原问题失真的情况）。
\end{itemize}

\subsubsection{B.6.3 基于问题物理量纲的无量纲调整建议}

为了使算法初始阶段具有较好的数值表现，建议参考问题的物理代价范围对初始 $\sigma$ 进行如下估算匹配：
\begin{equation}
\sigma_{\text{initial}} \approx \frac{\text{目标函数一阶成本波动范围}}{\text{状态决策变量的单步最大活动范围}}
\end{equation}

例如：若第一阶段目标函数的成本系数在 $[1, 100]$ 内变动，而跨阶段状态解耦变量 $z$ 的取值区间位于 $[0, 10]$ 之间，则推荐将各后续节点的初始惩罚阶数设为 $\sigma \approx 10$。

\subsection{B.7 机制形式化表达总结}

\begin{table}[htbp]
\centering
\caption{Sigma 自适应调整及对偶交互机制形式化总结}
\label{tab:sigma_summary}
\begin{tabular}{|c|c|p{7.5cm}|}
\hline
\textbf{控制核心参数} & \textbf{标准数学表达 / 逻辑分支} & \textbf{机制作用与控制本质} \\
\hline
初始设置规则 & $\sigma_1 = 0, \ \sigma_{t>1} = 1$ & 首阶段免除惩罚，其余节点默认保守初始化 \\
\hline
自适应放大公式 & $\sigma^{\text{new}} = \sigma^{\text{old}} \times \sigma_{\text{factor}}$ & 测试失败时按因子 $\sigma_{\text{factor}}=5$ 阶梯式放大 \\
\hline
对偶边界锁定 & $B_t = \sigma_t$ & 正则化系数直接等价赋予对偶乘子的无穷范数边界 \\
\hline
Sigma 测试条件 & $\|\text{obj}_{\text{reg}} - \text{obj}_{\text{non-reg}}\| \le \epsilon$ & 验证引入的近端范数项是否对原函数原解造成扭曲 \\
\hline
调整介入时机 & 外层割平面宏观满足收敛判定后 & 确保局部 Cut 逼近准确后，方启动 Sigma 足够性校验 \\
\hline
回滚影响链条 & 重置 \texttt{previous\_solution}，回退内层循环 & 消除由于模型曲率改变导致的失效收敛，重启探索 \\
\hline
\end{tabular}
\end{table}

\textbf{核心结论}：DynamicSDDiP.jl 引入的这套自适应 $\sigma$ 优化策略，其精妙之处在于\textbf{兼顾了正则化所需的乘子防振荡约束力与原问题解的逼近真度}。算法在初期使用较小的 $\sigma=1$ 来维持较高的求解精度，后期通过 Sigma 测试动态捕获对偶无界发散的风险。一旦检测到不一致，立即通过几何级数倍率（$\times 5$）放大 $\sigma$ 并锁定为对偶乘子空间的新屏障 $B_t$，同时敏锐地启动状态回滚重置，从而在多阶段随机规划的大规模反向 Cut 构造中完美平衡了数值稳定性和收敛速度。


\end{document}