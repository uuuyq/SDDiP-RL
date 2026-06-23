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

\subsection{1. 原始多阶段随机规划值函数}
在第 $t$ 阶段，给定前一阶段状态变量 $x_{t-1}$ 和当前场景的随机向量 $\xi_t$，最优值函数 $Q_t(x_{t-1}, \xi_t)$ 定义为：
\begin{equation}
Q_{t}(x_{t-1}, \xi_{t}) := \min_{x_{t}, y_{t}} \left\{ f_{t}(x_{t}, y_{t}) + \sum_{k \in K} p_{k} Q_{t+1}(x_{t}, \xi_{t+1}^{k}) : (x_{t}, y_{t}) \in \Phi_{n}(x_{t-1}, \xi_{t}) \right\}
\end{equation}
其中，点 $(x_{t-1}, \theta_{t}^{k})$ 构成了迭代计算中值函数上界的一个可行参考点。

\subsection{2. 引入状态解耦变量 $z_t$}
为了消除前后阶段状态变量的直接耦合，引入复制变量 $z_{t} = x_{t-1}$。未来阶段的期望值函数通过上方图（Epigraph）集合来约束逼近，问题可调整为如下等价形式：
\begin{align}
Q_{t}(x_{t-1}, \xi_{t}) = \min_{x_{t}, y_{t}, \theta_{t+1}^{k}} \quad & f_{t}(x_{t}, y_{t}) + \sum_{k \in K} p_{k} \theta_{t+1}^{k} \\
\text{s.t.} \quad & (x_{t}, y_{t}) \in \Phi_{n}(z_{t}, \xi_{t}) \\
& z_{t} = x_{t-1} \\
& \theta_{t+1}^{k} \ge Q_{t+1}(x_{t}, \xi_{t+1}^{k}), \quad \forall k \in K \\
& z_{t} \in Z_{t-1}, \quad x_{t} \in X_{t}
\end{align}

\subsection{3. 值函数的 Cut 逼近近似}
在迭代过程中，通过对未来所有场景依据概率 $p_k$ 取期望值，构造出与具体场景 $k$ 无关的统一状态逼近函数 $\psi_{t+1}^{i}(x_t)$：
\begin{equation}
\psi_{t+1}^{i}(x_t) = \max \left\{ \lambda_1, \lambda_2, \lambda_3, \dots, \lambda_i \right\}
\end{equation}
在第 $i+1$ 次迭代的前向过程中，用 $\psi_{t+1}^{i}(x_t)$ 替代真实的期望值函数，第 $t$ 阶段的近似子模型表述为：
\begin{align}
\mathcal{Q}_{t}^{i+1}(x_{t-1}, \xi_t) = \min_{x_t, y_t, \theta_{t+1}} \quad & f_{t}(x_{t}, y_{t}) + \theta_{t+1} \\
\text{s.t.} \quad & (x_{t}, y_{t}) \in \Phi_{n}(z_{t}, \xi_{t}) \\
& z_{t} = x_{t-1}^{i} \\
& \theta_{t+1} \ge \psi_{t+1}^{i}(x_{t}) \\
& z_{t} \in Z_{t-1}, \quad x_{t} \in X_{t}
\end{align}

\subsection{4. Epigraph 可行性判定子问题}
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

\subsection{1. 可行性问题的拉格朗日松弛与对偶割（Lagrangian Cut）的推导}
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

\subsection{2. 基于 Bundle 方法的对偶系统迭代求解}
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

\section{外层对偶割包络更新与级别集近端优化}

\subsection{1. 对偶值函数的线性割（Cut）更新表达式}
利用计算得到的次梯度，对偶值函数 $\omega_t^{i+1}(\pi, \pi_0)$ 的一阶泰勒支撑面（即对偶 Cut）表示为：
\begin{align}
\omega_t^{i+1}(\pi, \pi_0) &\le \omega_t^{i+1}(\hat{\pi}, \hat{\pi}_0) + \left(\frac{\partial \omega_t^{i+1}}{\partial \pi}\right)^\top (\pi - \hat{\pi}) + \frac{\partial \omega_t^{i+1}}{\partial \pi_0} (\pi_0 - \hat{\pi}_0) \nonumber \\
&= \omega_t^{i+1}(\hat{\pi}, \hat{\pi}_0) + \tilde{z}_t^\top (\pi - \hat{\pi}) + [f_t(\tilde{x}_t, \tilde{y}_t) + \tilde{\theta}_{t+1}] (\pi_0 - \hat{\pi}_0) \nonumber \\
&= \tilde{z}_t^\top \pi + [f_t(\tilde{x}_t, \tilde{y}_t) + \tilde{\theta}_{t+1}] \pi_0
\end{align}
注：最后一式利用了内层目标函数关于乘子的正齐次性结构（即 $\omega_t^{i+1}(\hat{\pi}, \hat{\pi}_0) = \hat{\pi}^\top \tilde{z}_t + \hat{\pi}_0 [f_t(\tilde{x}_t, \tilde{y}_t) + \tilde{\theta}_{t+1}]$）消除常数常项后展开得到。这与原文完全一致。

\subsection{2. 外层对偶最大化问题的割平面标准型}
在外层优化中，我们将所有历史生成的对偶 Cut 作为一个分段线性的上包络函数加入系统中：
\begin{align}
\max_{\pi, \pi_0} \quad & \omega_t^{i+1}(\pi, \pi_0) - \pi^\top x_{t-1}^i - \pi_0 \theta_t^i \\
\text{s.t.} \quad & \pi_0 \ge 0 \\
& (\pi, \pi_0) \in \mathcal{L}_1 \text{ 或 } \mathcal{L}_\infty \text{ 或 } \text{LN} \quad (\text{有界乘子约束空间}) \\
& \omega_t^{i+1}(\pi, \pi_0) \le \tilde{z}_t^\top \pi + [f_t(\tilde{x}_t, \tilde{y}_t) + \tilde{\theta}_{t+1}] \pi_0 \quad (\text{历史对偶 Cut 集合})
\end{align}
求解该最大化问题后，所得的最优目标函数值被记为当前系统对偶阶段的**上界（UB）**。
相应地，在当前最佳乘子点评估出来的实际松弛收益作为系统的稳定**下界（LB）**：
\begin{equation}
\text{LB} := \omega_t^{i+1}(\hat{\pi}, \hat{\pi}_0) - \hat{\pi}^\top x_{t-1}^i - \hat{\pi}_0 \theta_t^i
\end{equation}

\subsection{带有近端项的 Level（级别集）对偶步长更新}
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

\subsection{正则化策略的引入动机}

在上述的 Level Bundle 算法中，对偶乘子 $(\pi, \pi_0)$ 的搜索范围仅受限于规范化约束（如 $\mathcal{L}_1$ 或 $\mathcal{L}_\infty$），但缺乏对乘子绝对大小的限制。这可能导致以下问题：

\begin{itemize}
    \item 对偶问题可能无界，导致无法获得有效的割；
    \item 对偶乘子可能发散到无穷大，引发数值不稳定；
    \item 生成的割系数过大，影响后续迭代的收敛性。
\end{itemize}

为解决上述问题，引入正则化策略（Regularization），通过在目标函数中添加惩罚项，限制状态变量偏离固定值的程度，从而间接约束对偶乘子的大小。

\subsection{正则化项的数学形式}

在第 $t$ 阶段，给定固定的状态变量值 $x_{t-1}^i$（来自前向传播），正则化的内层拉格朗日松弛子问题修正为：

\begin{align}
\omega_t^{i+1}(\pi, \pi_0) = \min_{x_t, y_t, z_t, \theta_{t+1}} \quad & \pi_0 f_t(x_t, y_t) + \pi^\top z_t + \pi_0 \theta_{t+1} + \sigma_t \cdot \|z_t - x_{t-1}^i\| \\
\text{s.t.} \quad & (x_{t}, y_{t}) \in \Phi_{n}(z_{t}, \xi_{t}) \\
& \theta_{t+1} \ge \psi_{t+1}^{i}(x_{t}) \\
& z_{t} \in Z_{t-1}, \quad x_{t} \in X_{t}
\end{align}

其中：
\begin{itemize}
    \item $\sigma_t$ 为第 $t$ 阶段的正则化系数（Regularization Coefficient）；
    \item $\|\cdot\|$ 为选取的范数，通常为 $\ell_1$ 范数或 $\ell_\infty$ 范数。
\end{itemize}

\paragraph{$\ell_1$ 范数正则化：}

当选择 $\ell_1$ 范数时，正则化项为：

\begin{equation}
\|z_t - x_{t-1}^i\|_1 = \sum_{j=1}^{n_x} |z_{t,j} - x_{t-1,j}^i|
\end{equation}

$\ell_1$ 范数具有稀疏性诱导特性，倾向于生成稀疏的对偶乘子。

\paragraph{$\ell_\infty$ 范数正则化：}

当选择 $\ell_\infty$ 范数时，正则化项为：

\begin{equation}
\|z_t - x_{t-1}^i\|_\infty = \max_{j=1,\dots,n_x} |z_{t,j} - x_{t-1,j}^i|
\end{equation}

$\ell_\infty$ 范数具有均匀性特性，确保所有状态变量的偏离程度一致。

\subsection{3. 正则化系数与对偶边界的关系}

正则化系数 $\sigma_t$ 不仅用于惩罚状态变量的偏离，同时也作为对偶乘子的理论上界（Dual Bound）。

根据拉格朗日对偶理论，正则化项 $\sigma_t \cdot \|z_t - x_{t-1}^i\|$ 的系数 $\sigma_t$ 对应于对偶乘子 $\pi$ 的最大可能值。因此，我们有：

\begin{equation}
B_t = \sigma_t
\end{equation}

其中 $B_t$ 为第 $t$ 阶段的对偶边界值。

\paragraph{正则化策略配置：}

定义正则化策略配置 $\mathcal{R}_t = (\sigma_t, \|\cdot\|, \|\cdot\|_{\text{lifted}})$，其中：
\begin{itemize}
    \item $\sigma_t$：第 $t$ 阶段的正则化系数；
    \item $\|\cdot\|$：前向传播中使用的正则化范数；
    \item $\|\cdot\|_{\text{lifted}}$：后向传播中使用的范数，用于范数边界约束。
\end{itemize}

\paragraph{正则化系数的自适应调整：}

在迭代过程中，$\sigma_t$ 可根据算法收敛情况进行调整。定义缩放因子 $\sigma_{\text{factor}} > 0$，当算法收敛时，通过下式验证解的质量：

\begin{equation}
\sigma_t^{\text{new}} = \sigma_{\text{factor}} \cdot \sigma_t
\end{equation}



\section{范数边界约束的数学形式}

范数边界约束（Norm Bound Constraints）直接限制对偶乘子 $(\pi, \pi_0)$ 的大小，确保对偶问题有唯一解且数值稳定。

\subsection{约束的基本形式}

范数边界约束的核心思想是限制 $\pi$ 相对于 $\pi_0$ 的比值：

\begin{equation}
\frac{\|\pi\|}{\pi_0} \le B_t
\end{equation}

其中 $B_t$ 为对偶边界值。

\subsection{$\ell_1$ 范数边界约束（限制 $\ell_\infty$ 范数）}

当选择 $\|\cdot\|_{\text{lifted}} = \ell_1$ 时，范数边界约束具体化为：

\begin{equation}
|\pi_j| \le B_t \cdot w_j \cdot \pi_0, \quad \forall j = 1,\dots,n_x
\end{equation}

其中 $w_j$ 为第 $j$ 个状态变量的权重系数。

\textbf{等价的 $\ell_\infty$ 范数约束：}

\begin{equation}
\|\pi\|_\infty \le B_t \cdot \|w\|_\infty \cdot \pi_0
\end{equation}

\subsection{ $\ell_\infty$ 范数边界约束（限制加权 $\ell_1$ 范数）}

当选择 $\|\cdot\|_{\text{lifted}} = \ell_\infty$ 时，范数边界约束具体化为：

\begin{equation}
\sum_{j=1}^{n_x} w_j \cdot |\pi_j| \le B_t \cdot \pi_0
\end{equation}

\textbf{等价的加权 $\ell_1$ 范数约束：}

\begin{equation}
\|\pi\|_{w,1} = \sum_{j=1}^{n_x} w_j \cdot |\pi_j| \le B_t \cdot \pi_0
\end{equation}

\subsection{约束的几何解释}

范数边界约束在 $(\pi_0, \pi)$ 空间中定义了一个有界区域：

\begin{itemize}
    \item $\ell_1$ 范数边界约束定义了一个\textbf{棱锥}：所有对偶变量必须满足 $|\pi_j| \le B_t \cdot w_j \cdot \pi_0$；
    \item $\ell_\infty$ 范数边界约束定义了一个\textbf{多面体锥}：所有对偶变量必须满足 $\sum_{j=1}^{n_x} w_j \cdot |\pi_j| \le B_t \cdot \pi_0$。
\end{itemize}

\section{权重系数的计算}

权重系数 $w_j$ 用于缩放不同状态变量的对偶乘子边界，反映状态变量的相对重要性。

\section{二进制近似下的权重计算}

当状态变量采用二进制近似（Binary Approximation）时，权重系数由下式计算：

\begin{equation}
w_j = 2^{k_j - 1} \cdot \beta_j
\end{equation}

其中：
\begin{itemize}
    \item $k_j$：状态变量 $x_j$ 的二进制分解位数；
    \item $\beta_j$：状态变量 $x_j$ 的二进制精度（每个二进制位代表的数值）。
\end{itemize}

\subsection{权重系数的含义}

在二进制近似中，原始整数状态 $x_j$ 被分解为：

\begin{equation}
x_j = \sum_{m=1}^{k_j} 2^{m-1} \cdot \beta_j \cdot b_{j,m}
\end{equation}

其中 $b_{j,m}$ 为二进制变量（0 或 1）。权重系数 $w_j$ 反映了二进制位对原始状态的贡献程度：
\begin{itemize}
    \item 高位（$m$ 大）的权重更大，因为 $2^{m-1}$ 更大；
    \item 精度 $\beta_j$ 越高，权重越大。
\end{itemize}

\subsection{非二进制近似下的权重}

当不使用二进制近似（No State Approximation）时，权重系数默认为 1：

\begin{equation}
w_j = 1, \quad \forall j = 1,\dots,n_x
\end{equation}

\section{规范化约束与范数边界约束的协同作用}

在 Outer Problem 中，规范化约束和范数边界约束同时生效，共同定义对偶乘子的可行域。

\subsection{ 完整的约束体系}

以选择 $\mathcal{L}_1$ 规范化和 $\ell_1$ 范数边界约束为例，完整的约束体系为：

\begin{align}
\max_{\pi, \pi_0} \quad & \omega_t^{i+1}(\pi, \pi_0) - \pi^\top x_{t-1}^i - \pi_0 \theta_t^i \\
\text{s.t.} \quad & \pi_0 \ge \epsilon \quad (\text{非负约束，}\epsilon > 0) \\
& \sum_{j=1}^{n_x} |\pi_j| + \pi_0 \le 1 \quad (\mathcal{L}_1 \text{ 规范化约束}) \\
& |\pi_j| \le B_t \cdot w_j \cdot \pi_0, \quad \forall j \quad (\ell_1 \text{ 范数边界约束}) \\
& \omega_t^{i+1}(\pi, \pi_0) \le \tilde{z}_t^\top \pi + [f_t(\tilde{x}_t, \tilde{y}_t) + \tilde{\theta}_{t+1}] \pi_0 \quad (\text{历史对偶 Cut})
\end{align}

\subsection{ 约束的作用分工}

\begin{itemize}
    \item \textbf{规范化约束}：定义对偶变量的相对比例（单位球），确保生成"最深"的割；
    \item \textbf{范数边界约束}：限制对偶变量的绝对大小，防止数值发散；
    \item \textbf{两者协同}：规范化约束定义可行域的形状，范数边界约束定义可行域的大小，共同确保对偶问题有唯一解且数值稳定。
\end{itemize}

\subsection{ 可行域的几何描述}

最终可行域是规范化约束的单位球与范数边界约束的锥的交集：

\begin{equation}
\mathcal{F} = \{ (\pi, \pi_0) \mid \|\tilde{\pi}\| \le 1 \} \cap \{ (\pi, \pi_0) \mid \frac{\|\pi\|}{\pi_0} \le B_t \cdot w \}
\end{equation}

其中 $\tilde{\pi} = (\pi_0, \pi_1, \dots, \pi_{n_x})$。

\section{ 正则化与范数边界约束的算法流程集成}

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