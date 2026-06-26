# Deep Lagrangian Cut 的 Level Bundle 求解算法

## 1. 原始 Deep Cut 对偶问题

给定 trial point

\[
(X^{\text{trial}},\theta^{\text{trial}})
\]

定义 Lagrangian 子问题值函数

\[
L(\pi,\pi_0)
=
\min_{(\lambda,z)\in W}
\left\{
\pi^\top z
+
\pi_0\,Q(\lambda)
\right\}.
\]

Deep Cut 对应的分离问题为

\[
\max_{\pi,\pi_0}
\;
L(\pi,\pi_0)
-
\pi^\top X^{\text{trial}}
-
\pi_0 \theta^{\text{trial}}
\]

满足归一化约束

\[
\|(\pi,\pi_0)\| \le 1.
\]

---

## 2. 本实现采用的归一化

代码中使用加权 \(L_1\) norm：

\[
\sum_{i=1}^{n}
|\pi_i|
+
20\,\pi_0
\le 1,
\]

且

\[
\pi_0 \ge 0.
\]

因此实际求解的是

\[
\max_{\pi,\pi_0}
\;
L(\pi,\pi_0)
-
\pi^\top X^{\text{trial}}
-
\pi_0 \theta^{\text{trial}}
\]

s.t.

\[
\sum_i |\pi_i|
+
20\pi_0
\le 1,
\]

\[
\pi_0 \ge 0.
\]

这实际上是论文中的

Weighted \(L_1\)-Deep Cut。

---

# 3. Bundle Master Problem

由于

\[
L(\pi,\pi_0)
\]

不可直接表示，

利用 Bundle 方法构造其切平面近似。

---

## 3.1 Inner Problem

给定当前中心点

\[
(\hat\pi^k,\hat\pi_0^k),
\]

求解

\[
L(\hat\pi^k,\hat\pi_0^k)
=
\min_{(\lambda,z)\in W}
\Big(
(\hat\pi^k)^\top z
+
\hat\pi_0^k Q(\lambda)
\Big).
\]

得到最优解

\[
(z^k,q^k).
\]

其中

\[
q^k = Q(\lambda^k).
\]

---

## 3.2 次梯度

由 Lagrangian Dual 理论

\[
g^k
=
\begin{bmatrix}
z^k \\
q^k
\end{bmatrix}
\]

是

\[
L(\pi,\pi_0)
\]

在

\[
(\hat\pi^k,\hat\pi_0^k)
\]

处的一个次梯度。

---

## 3.3 Bundle Cut

构造线性上界

\[
L
\le
(z^k)^\top \pi
+
q^k \pi_0.
\]

每次迭代向 Master 中加入

\[
L
\le
(z^j)^\top \pi
+
q^j \pi_0,
\qquad
j=1,\dots,k.
\]

---

# 4. Outer Master Problem

当前 Bundle 近似为

\[
\hat L_k(\pi,\pi_0)
=
\min_j
\{
(z^j)^\top\pi
+
q^j\pi_0
\}.
\]

于是 Master Problem 为

\[
\max_{L,\pi,\pi_0}
\;
L
-
\pi^\top X^{\text{trial}}
-
\pi_0 \theta^{\text{trial}}
\]

s.t.

\[
L
\le
(z^j)^\top\pi
+
q^j\pi_0,
\qquad
j=1,\dots,k
\]

\[
\sum_i |\pi_i|
+
20\pi_0
\le 1
\]

\[
\pi_0 \ge 0.
\]

得到

\[
UB_k.
\]

---

# 5. Lower Bound

利用当前中心点得到

\[
LB_k
=
L(\hat\pi^k,\hat\pi_0^k)
-
(\hat\pi^k)^\top X^{\text{trial}}
-
\hat\pi_0^k
\theta^{\text{trial}}.
\]

更新

\[
LB
=
\max
\{
LB,
LB_k
\}.
\]

并记录对应最优乘子

\[
(\pi^\star,\pi_0^\star).
\]

---

# 6. Level Bundle Step

定义

\[
UB = UB_k,
\]

\[
LB = \max_{j\le k} LB_j.
\]

Gap

\[
\Delta_k
=
UB-LB.
\]

构造 level

\[
\ell_k
=
UB
-
\alpha
(UB-LB),
\]

其中

\[
\alpha
=
\texttt{level\_factor}.
\]

通常

\[
\alpha\in(0,1).
\]

---

## Level Constraint

要求下一次迭代点满足

\[
L
-
\pi^\top X^{\text{trial}}
-
\pi_0\theta^{\text{trial}}
\ge
\ell_k.
\]

---

## Proximal Subproblem

以当前中心

\[
(\hat\pi^k,\hat\pi_0^k)
\]

为参考，

求解

\[
\min_{\pi,\pi_0}
\;
\|\pi-\hat\pi^k\|_2^2
+
(\pi_0-\hat\pi_0^k)^2
\]

s.t.

\[
L
\le
(z^j)^\top\pi
+
q^j\pi_0,
\qquad
j=1,\dots,k
\]

\[
L
-
\pi^\top X^{\text{trial}}
-
\pi_0\theta^{\text{trial}}
\ge
\ell_k
\]

\[
\sum_i |\pi_i|
+
20\pi_0
\le 1
\]

\[
\pi_0\ge0.
\]

得到新的中心

\[
(\hat\pi^{k+1},\hat\pi_0^{k+1}).
\]

---

# 7. 收敛判据

若

\[
UB-LB
<
\varepsilon |UB|
\]

或

\[
UB-LB
<
10^{-6},
\]

则认为 Bundle 收敛。

---

# 8. Deep Lagrangian Cut 恢复

最终得到

\[
(\pi^\star,\pi_0^\star).
\]

重新求解一次 Inner Problem 得到

\[
L(\pi^\star,\pi_0^\star).
\]

若

\[
\pi_0^\star >0,
\]

则生成 Deep Lagrangian Cut

\[
\theta
\ge
\frac{L(\pi^\star,\pi_0^\star)}{\pi_0^\star}
-
\frac{(\pi^\star)^\top}{\pi_0^\star}x.
\]

记

\[
\beta
=
\frac{L(\pi^\star,\pi_0^\star)}
{\pi_0^\star},
\]

\[
g
=
-\frac{\pi^\star}
{\pi_0^\star},
\]

最终切平面形式为

\[
\boxed{
\theta
\ge
\beta
+
g^\top x
}
\]

即为加入 SDDiP 的 Deep Lagrangian Cut。