# skill-optimizer Specification

## Purpose
TBD - created by archiving change implement-greater-skill-optimizer. Update Purpose after archive.
## 需求
### 需求：GREATER 优化流程
系统必须实现基于 "Gradient Over Reasoning" 的四阶段迭代优化流程，以利用小模型自身梯度优化 Skill 提示词。

#### 场景：端到端优化
当用户启动优化任务时，系统应按照顺序自动执行候选提案、推理生成、梯度计算和选择更新四个阶段，直到完成指定的迭代次数。

---

## 修改需求
### 需求：第一阶段 - 候选词提案 (Candidate Proposal)
系统必须通过小模型的前向概率筛选候选 Token，使用更鲁棒的交集策略。
- **输入**: 当前 Prompt 前缀、Batch 样本数据。
- **流程**:
    1. **Logits 计算**: 对 Batch 中每个样本，计算当前 Prompt 位置 $p_i$ 的下一个 Token 概率分布。
    2. **Top-k 筛选**: 选取每个样本概率最高的 Top-k 个 Token。
    3. **鲁棒交集聚合**: 采用 **"Union of Intersections"** 策略：即多次随机采样 Logits 子集计算交集，最后取这些交集的并集，以确保在 Batch 较大或差异较大时仍有足够候选。
    4. **稀疏嵌入**: 仅构建候选词对应的 One-Hot 指示向量 $\epsilon_i$，用于后续高效梯度计算。

#### 场景：筛选候选
在优化的每一步，系统首先通过 Logits 和鲁棒交集操作筛选出即符合当前上下文又具备通用性的候选词。

---

### 需求：第三阶段 - 基于推理的梯度计算 (Gradient Over Reasoning)
系统必须通过反向传播计算候选词的梯度，衡量其对正确推理的贡献。
- **输入**: 完整上下文、Ground Truth $y$。
- **流程**:
    1. **计算损失**: 计算预测 $\hat{y}$ 与真实标签 $y$ 的目标损失 $L_{target}$。
    2. **控制损失 (可选)**: 若配置了 `control_weight`，需计算 Prompt 自身的控制损失 $L_{control}$ (如正则化或平滑项)，总损失 $L = L_{target} + \lambda L_{control}$。
    3. **反向传播**: 计算 $L$ 相对于第一阶段生成的**候选词指示向量 $\epsilon_i$** 的梯度 $\frac{\partial L}{\partial \epsilon_i}$。

#### 场景：获取指导信号
核心梯度不仅仅反映答案对错，更通过推理链回传，反映了该 Token 是否有助于引导出"正确的推理路径"，同时兼顾 Prompt 的流畅性。

## 新增需求
### 需求：Conversation Template 支持
系统必须支持主流微调模型的对话模板（如 Llama-2, Llama-3 格式），以确保优化过程中的 Prompt 结构与模型预训练格式一致。
- **功能**:
    - 支持 `llama-2`, `llama-3` 等常见模板。
    - 能够精确识别 Prompt 中的 User/Assistant 角色、Goal、Control、Target 等部分的 Token Slice 范围。
    - 支持在 Template 环境下动态更新 Control 部分的 Token。

#### 场景：Template 适配
优化器在构建 Input IDs 时，自动应用配置的 Template，并根据 Template 结构动态计算用于 Loss 和 Gradient 的 Slice 索引。

