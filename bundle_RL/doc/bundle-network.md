

# 实验配置
Multi-Commodity Network Design (MC)
Generalized Assignment (GA)

都是 MILP
都可以做 Lagrangian Relaxation
一个是等式约束，一个是不等式约束


# 其他的一些尝试
- 训练时固定 unrolling horizon = 10，泛化到其他iteration，即使只训练 10 步，在 50/100 steps 仍然有效

- sampling or no sampling
  - no sampling  网络学的是 x→best update
  - 同一个输入：永远得到同一个 hidden state。因此：attention 固定 update 固定 trajectory 固定
  - sampling  网络学的是 x→distribution over updates  
  - 网络输出的是概率分布，从概率分布中采样出下一个阶段的update；推理时不采样，直接使用均值（推理阶段要的是稳定收敛，不是探索。）
  - 
- 模型输出使用softmax、Sparsemax做了对比，一种是密集型，一种是稀疏型

# 后面需要考虑的问题
- baseline 还是用了Gradient Ascent、Adam，如果 2 iterations 没 improvement：step size /= 2
- bundle-network使用了bundle算法的变体，并使用了网格搜索对bundle算法的超参数进行优化，使用最优参数作为baseline




# 实验
- 训练步数
- 特征输入 （x_t-1、xi、stage、iteration、是否需要添加
- 训练的数据量
- 网络的维度





