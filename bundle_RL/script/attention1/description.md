# Attention RL 设置

**特征提取器**: `AttentionBundleExtractor`
- **结构**: 基于 Transformer 的编码器架构
- **组件**: CutEncoder + Self-Attention + GlobalEncoder
- **核心**: 利用注意力机制建模 cuts 之间的依赖关系
- **输出**: 128 维特征向量（含 cls_embedding）

**PPO 策略**:
- `MultiInputActorCriticPolicy` + 注意力特征提取器
- Actor/Critic 共享编码器参数
- Actor/Critic 网络结构: `[128, 128]`

**特点**: 通过自注意力机制捕捉 cuts 的全局依赖，提升策略表达能力。