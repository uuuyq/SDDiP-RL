# 在 default_feature 上进一步调整
- 不需要输出 eta 步长，步长使用正常的 bundle 算法提供的步长，
  - serious step 判断可以进行步长的调整，但是如果要使用 RL，完全不用 Master 求解，就没有 ub 了，所以也不能使用 serious step 判断来调整步长了
  - **那就使用固定的步长 0.5**

- 增加 state，即增加输入的特征
  - valid_mask
    state 中维护一个 mask 向量，记录哪些 cuts 是 padding 的，后面 step 中计算 lambda 时可以使用 mask，先将 padding 的 action 输出值 mask 掉
    ，应该不需要作为 RL model 的输入
  - Search Direction Norm
    上一次得到的增量 d 的强度，使用 d 的二范数
  - Linear Improvement
    cut 的截距项的加权和，使用 lambda 作为权重，反应 master 线性部分的目标值
  - cosine(cut, d)
    新 cut 与 d 的夹角，使用余弦相似度


state = [
    cuts,
    valid_mask,
    search_direction_norm,
    linear_improvement,
    cosine(cut, d)
]


模型使用 default_feature 相同的模型规模，训练以及测试数据的代码风格保持一致，数据逻辑保持一致
