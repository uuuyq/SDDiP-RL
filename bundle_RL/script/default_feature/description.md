# Default Feature RL 设置

**特征提取器**: `SimpleBundleExtractor`
- **结构**: 两层全连接网络（256 → 128）
- **输入**: 拼接 cuts、pi、trial_point、realization 特征
- **输出**: 128 维特征向量

**PPO 策略**:
- `MultiInputActorCriticPolicy` + 自定义特征提取器
- Actor/Critic 网络结构: `[128, 128]`

**特点**: 简单直接的特征拼接方式，适合作为基线对比。