# New Lagrangian Cut Generation Framework (Füllner et al.)

## 核心思想

传统 SDDiP 中：

对于节点 n 的价值函数

Q_n(x_{a(n)})

通过求解 Lagrangian Dual

max_{π_n}
    L_n(π_n) + π_n^T x_{a(n)}

得到 Lagrangian cut

θ_n ≥ L_n(π_n) + π_n^T x_{a(n)}

但该对偶问题通常退化严重：

- 存在大量最优对偶解
- 不同 cut 质量差异巨大
- 容易生成非常陡峭的 cut
- 求解效率较低

---

## Step 1：从 Value Function 转换为 Epigraph Separation

定义价值函数上图(epigraph)

epi(Q_n)
=
{
(x, θ) :
θ ≥ Q_n(x)
}

传统方法：

直接逼近 Q_n(x)

新框架：

直接逼近

epi(Q_n)

这样 Optimality Cut 与 Feasibility Cut 可以统一处理。

---

## Step 2：构造 Feasibility Problem

给定 trial point

(x_a^i, θ_n^i)

判断其是否属于当前近似 epigraph：

min 0

s.t.

(λ_n, z_n) ∈ W_n

z_n = x_a^i

θ_n^i ≥ c_n^T λ_n

若可行：

    (x_a^i, θ_n^i)
    ∈ epi(Q_n)

否则：

    需要生成 cut

---

## Step 3：Lagrangian Relaxation

松弛

z_n = x_a^i

以及

θ_n^i ≥ c_n^T λ_n

引入乘子

π_n
π_n^0 ≥ 0

得到

L_n(π_n, π_n^0)

=
min

π_n^T z_n

+
π_n^0 c_n^T λ_n

s.t.

(λ_n,z_n)∈W_n

---

## Step 4：构造新的 Lagrangian Dual

得到统一框架下的 Dual

max

L_n(π_n,π_n^0)

− π_n^T x_a^i

− π_n^0 θ_n^i

该问题具有性质：

如果

(x_a^i, θ_n^i)

不在

epi(co(Q_n))

中

则 Dual 无界。

因此需要归一化(normalization)。

---

# Step 5：Normalization

论文核心创新

## 方法1：Norm Normalization

加入

g(π_n,π_n^0)
=
||(π_n,π_n^0)||

≤ 1

求解

max

L_n(π_n,π_n^0)

− π_n^T x_a^i

− π_n^0 θ_n^i

s.t.

||(π_n,π_n^0)|| ≤ 1

得到

Deep Lagrangian Cut

---

### 几何意义

等价于

寻找距离 trial point

最远的支持超平面

即

Maximum Depth Cut

---

### 常见选择

ℓ2-deep

ℓ1-deep

ℓ∞-deep

Weighted ℓ1-deep

---

## 方法2：Linear Normalization

选择一个 core point

(u_n,u_n^0)

加入

u_n^T π_n
+
u_n^0 π_n^0

≤ 1

求解

max

L_n(π_n,π_n^0)

− π_n^T x_a^i

− π_n^0 θ_n^i

s.t.

u_n^T π_n
+
u_n^0 π_n^0

≤ 1

得到

LN Cut
(Linear Normalization Cut)

---

### 几何意义

沿方向

(u_n,u_n^0)

从 trial point

投影到

epi(co(Q_n))

上

得到支持超平面

---

### 理论性质

若 core point 合适：

LN Cut

通常

- Facet-defining
- Pareto-optimal

---

# Step 6：生成 Cut

得到最优乘子

(π_n^*, π_n^{0*})

生成 cut

π_n^{0*} θ_n

+
(π_n^*)^T x_a

≥

L_n(π_n^*, π_n^{0*})

加入

Ψ_n

(当前 epigraph approximation)

---

# Step 7：嵌入 SDDiP

Backward Pass：

for each node n:

    solve normalized dual

    generate cut

    add cut to Ψ_n

Forward Pass：

    solve policy using updated cuts

重复直到收敛

---

# 整体伪代码

Initialize Ψ_n

while not converged:

    # Forward pass
    simulate scenario path

    obtain trial states

        x_a^i

    # Backward pass
    for stages T→1:

        for each node n:

            compute trial point

                (x_a^i, θ_n^i)

            solve

                Normalized Dual

            obtain

                (π_n, π_n^0)

            construct

                Lagrangian Cut

            add cut to Ψ_n

return policy

---

# 不同 Cut 的关系

Classical Lagrangian Cut
    ↓
Normalized Dual

    ├─ ℓ2 Normalization
    │      → Deep Cut
    │
    ├─ ℓ1 Normalization
    │      → ℓ1-Deep Cut
    │
    ├─ ℓ∞ Normalization
    │      → ℓ∞-Deep Cut
    │
    └─ Linear Normalization
           → LN Cut
           → Facet-defining
           → Pareto-optimal

---

# 论文最终结论

在 SDDiP 中：

Deep Cut 和 LN Cut

相比

- Classical Lagrangian Cut
- Benders Cut

能够显著提升 Lower Bound。

其中：

LN Cut 表现最好，

尤其在二进制状态空间下能够更快逼近
conv(Q_n) 的真实 facet。

但代价是：

- 对偶问题更难求
- Core point 选择敏感
- 数值稳定性要求更高
- 计算时间增加