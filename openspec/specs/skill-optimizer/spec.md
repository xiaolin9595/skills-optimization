# skill-optimizer Specification

## Purpose
TBD - created by archiving change implement-greater-skill-optimizer. Update Purpose after archive.
## 需求
### 需求：GREATER 优化流程
系统必须实现基于 "Gradient Over Reasoning" 的四阶段迭代优化流程，以利用小模型自身梯度优化 Skill 提示词。

#### 场景：端到端优化 (Refined)
当用户启动优化任务时，系统应按照以下逻辑顺序执行：
1. **生成推理 (Reasoning Generation)**: 基于当前 Control 生成推理链 $r$。
2. **构建上下文 (Context Construction)**: 将生成的推理链 $r$ 和提取提示词 $p_{extract}$ 插入到 Prompt 中。
3. **候选提案 (Candidate Proposal)**: 基于前缀 Logits 筛选候选词。
4. **梯度计算 (Gradient Computation)**: 基于完整的 `Prompt + Reasoning + Extract + Target` 上下文计算梯度。
5. **选择更新**: 选择最佳候选并更新。

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

### 需求：第二阶段 - 推理生成与答案提取 (Reasoning Generation)
系统必须强制模型生成推理过程，而非直接预测答案。

#### 场景：接口鲁棒性 (New)
`generate_reasoning` 接口必须支持批量输入（Batch Processing）并提供灵活的输出格式控制。
- **批量支持**: 必须接受 `input_ids [Batch, Seq]` 作为输入。
- **解码控制**: 必须通过参数（如 `decode=True`）控制返回解码后的文本列表（用于 Prompt 注入）还是原始 Token IDs（用于计算梯度或拼接）。
- **用途分离**:
    - 在 **Optimization Loop** 中，系统调用该接口获取 **Text** 以更新 `SkillPrompter`。
    - 在 **Selection/Validation** 中，系统调用该接口获取 **IDs** 以进行快速的 Loss 计算拼接。

### 需求：第三阶段 - 基于推理的梯度计算 (Gradient Over Reasoning)
系统必须通过反向传播计算候选词的梯度，衡量其对正确推理的贡献。
- **流程**:
    1. **计算损失**: 
        - **Target Loss**: 标准交叉熵。
        - **Focused Loss**: 若定义了 `focused_target`，系统必须使用 Sliding Window (Unfolding) 机制计算生成文本中与 `focused_target` 最匹配片段的 Loss，以捕捉关键词是否出现。
        - **Control Loss**: 可选的控制稳定性 Loss。
    2. **梯度计算**: 计算总 Loss 相对于候选词的梯度。
    3. **归一化**: 在聚合多个样本的梯度前，系统必须对**每个样本**的梯度向量进行 **L2 归一化**，防止长文本或高 Loss 样本主导梯度方向。

#### 场景：获取鲁棒梯度
核心梯度不仅仅反映答案对错，更通过推理链回传，反映了该 Token 是否有助于引导出"正确的推理路径"，同时通过归一化和 Focused Loss 确保在复杂场景下的鲁棒性。

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

### 需求：Conversation Template 支持
系统必须支持主流微调模型的对话模板（如 Llama-2, Llama-3 格式），以确保优化过程中的 Prompt 结构与模型预训练格式一致。
- **功能**:
    - 支持 `llama-2`, `llama-3` 等常见模板。
    - **Llama-3 支持**: 必须针对 Llama-3 的特殊标记（如 `<|start_header_id|>`, `<|eot_id|>`）实现精确的手动构建和 Slicing 逻辑。
    - 能够精确识别 Prompt 中的 User/Assistant 角色、Goal、Control、Target 等部分的 Token Slice 范围。
    - 支持在 Template 环境下动态更新 Control 部分的 Token。

#### 场景：Llama-3 适配
当配置 `template_name="llama-3"` 时，系统生成符合 Llama-3 Instruct 格式的 Prompt，并正确计算 Assistant 回复部分的 Loss Slice，避开 Header 标记。

### 需求：动态控制长度与搜索策略 (Dynamic Control Strategy)
系统必须支持动态调整 Control Prompt 的长度，以平衡搜索空间和表达能力。

#### 场景：长度初始化与强制对齐 (New)
在开始优化前，系统必须将初始 Skill Content 的长度调整为配置的 `start_len`。
- **截断**: 若输入 Content 长度超过 `start_len`，系统必须将其截断，保留前 `start_len` 个 Token。
- **填充**: 若输入 Content 长度不足，系统必须使用占位符（如 `!`）填充至 `start_len`。
- **目的**: 确保优化过程严格遵循 `start_len` 到 `end_len` 的维度递增逻辑，避免因初始 Prompt 过长导致的维度错配和资源浪费。

#### 场景：递增增长
当 `length_iter` 增加时，系统必须在当前最优 Control 后追加**正好一个**占位符 Token，确保搜索空间维度精确加一。

### 需求：候选词过滤 (Token Filtering)
系统必须确保生成的 Control Prompt 具有可读性并且不包含特殊字符。
- **Non-ASCII 过滤**: 在 Stage 1 (Candidate Proposal) 计算 Logits 时，系统必须将 Non-ASCII Token 和不可打印字符的概率设为 `-inf`，仅允许 ASCII 字符候选。

#### 场景：合法性约束
在生成候选词时，系统自动屏蔽乱码或不可见字符，确保优化得到的 Prompt 是人类可读的英文文本。

### 需求：模型适配增强
系统必须针对不同模型架构（如 Llama-2, Llama-3, Falcon）正确配置 Tokenizer 的 Padding Side, Pad Token, BOS/EOS Token，并必须使用正确的 Slice Offset（如 Llama-2 的 `-3` Loss Slice 修正）以对齐模型内部机制。

#### 场景：模型兼容性
当切换到 Llama-2 模型时，系统自动识别并应用 `-3` 的 Loss Slice 偏移量，以规避 EOS Token 对梯度的干扰。

### 需求：数据集驱动优化 (Dataset-Driven Optimization)
系统必须支持从外部数据源加载训练样本，并能自动识别多种常用的 Benchmark 数据格式。

#### 场景：BBH (Big Bench Hard) 数据加载 (New)
当加载以 `.json` 结尾但内容为 CSV 格式（标题行包含 `goal,target,final_target`）的 BBH 数据文件时，加载器必须能够正确解析逗号分隔符和引号包裹的文本字段。
- **字段映射**:
    - `goal` -> 任务目标/问题。
    - `target` -> 答案选项或标签。
    - `final_target` -> 提取答案所需的关键词。

#### 场景：官方 BBH 初始提示词对齐 (New)
在使用 BBH 数据集进行优化时，系统应能够配置并使用 GreaTer 官方推荐的初始提示词（Control Initial），例如 `" proper logical reasoning and think step by step. Finally give the actual correct answer."`，以确保实验起点的一致性。
- **长度验证**: 系统必须自动验证初始提示词在目标模型 Tokenizer 下的 Token 数量，并根据配置的 `start_len` 进行自动裁剪或填充。

### 需求：系统可观测性 (System Observability)
系统必须提供详细的执行追踪，以便在复杂优化过程中验证算法的正确性。

#### 场景：梯度与候选词追踪 (Candidate Trace)
在每一轮迭代中，系统必须记录并输出：
- **候选词质量**: 提出的候选词数量及其解码后的文本。
- **梯度分布**: 聚合梯度的最小值、最大值和均值。
- **选择逻辑**: 被选中进行验证的 Top-$\mu$ 个词以及它们的实测 Loss 值。
- **Focused Loss 对齐**: 记录 Unfolding 窗口选择的最佳匹配偏移量，验证系统是否成功对齐到目标答案。

#### 场景：日志持久化
实验脚本必须支持将所有 `INFO` 级别以上的日志持久化到指定文件，以便在大规模训练时进行离线分析。

