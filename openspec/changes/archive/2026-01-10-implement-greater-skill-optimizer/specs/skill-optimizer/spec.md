# 规范：技能优化器实现

## 新增需求

### 需求：GREATER 优化流程
系统必须实现基于 "Gradient Over Reasoning" 的四阶段迭代优化流程，以利用小模型自身梯度优化 Skill 提示词。

#### 场景：端到端优化
当用户启动优化任务时，系统应按照顺序自动执行候选提案、推理生成、梯度计算和选择更新四个阶段，直到完成指定的迭代次数。


---

### 需求：第一阶段 - 候选词提案 (Candidate Proposal)
系统必须通过小模型的前向概率筛选候选 Token，而非在全词表上计算。
- **输入**: 当前 Prompt 前缀、Batch 样本数据。
- **流程**:
    1. **Logits 计算**: 对 Batch 中每个样本，计算当前 Prompt 位置 $p_i$ 的下一个 Token 概率分布。
    2. **Top-k 筛选**: 选取每个样本概率最高的 Top-k 个 Token。
    3. **交集聚合**: 取 Batch 内所有样本候选词的**交集**，得到通用候选集 $candidates_i$。
    4. **稀疏嵌入**: 仅构建候选词对应的 One-Hot 指示向量 $\epsilon_i$，用于后续高效梯度计算。

#### 场景：筛选候选
在优化的每一步，系统首先通过 Logits 和交集操作快速缩小搜索范围，避免无效计算。

---

### 需求：第二阶段 - 推理生成与答案提取 (Reasoning Generation)
系统必须强制模型生成推理过程，而非直接预测答案。
- **流程**:
    1. **生成推理链**: 将 `[Prompt] + [Input]` 输入模型，采样生成推理文本 $r$ (Chain-of-Thought)。
    2. **引导提取**: 拼接固定提取提示词 $p_{extract}$ (如 "Therefore, the final answer is")，引导模型输出最终答案 Logits $\hat{y}$。

#### 场景：构建上下文
系统构建包含完整推理路径的上下文 `[Prompt] + [Input] + [Reasoning Chain] + [Extract Prompt]`，确保 Loss 包含推理过程信息。

---

### 需求：第三阶段 - 基于推理的梯度计算 (Gradient Over Reasoning)
系统必须通过反向传播计算候选词的梯度，衡量其对正确推理的贡献。
- **输入**: 完整上下文、Ground Truth $y$。
- **流程**:
    1. **计算损失**: 计算预测 $\hat{y}$ 与真实标签 $y$ 的交叉熵损失 $L$ (可结合困惑度正则化)。
    2. **反向传播**: 计算 $L$ 相对于第一阶段生成的**候选词指示向量 $\epsilon_i$** 的梯度 $\frac{\partial L}{\partial \epsilon_i}$。

#### 场景：获取指导信号
核心梯度不仅仅反映答案对错，更通过推理链回传，反映了该 Token 是否有助于引导出"正确的推理路径"。

---

### 需求：第四阶段 - 候选词选择与更新 (Selection & Update)
系统必须基于梯度和实测 Loss 择优更新 Prompt。
- **流程**:
    1. **梯度筛选**: 选择负梯度最大（即最能降低 Loss）的 Top-$\mu$ 个候选词。
    2. **实测验证**: 将优选词填入 Prompt，在训练集上执行前向传播，计算实际 Loss。
    3. **更新**: 比较实际 Loss，若优于当前最优值，则更新 Prompt 位置 $p_i$。

#### 场景：迭代优化
系统按顺序对 Prompt 的每个 Token 位置执行上述四个阶段，循环迭代直到收敛或达到最大步数。

---

### 需求：参考实现对齐
具体实现必须参考 `referenceSolution/GreaTer` 代码库中的逻辑和细节，确保算法行为一致。

#### 场景：代码移植
在实现核心算法逻辑时，开发者应查阅 `referenceSolution/GreaTer` 目录下的 Python 源码（如 `llm_opt/gcg/greater_opt.py`）作为主要参考依据。
