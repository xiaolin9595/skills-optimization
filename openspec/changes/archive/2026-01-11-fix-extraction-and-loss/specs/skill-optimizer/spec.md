## 修改需求

### 需求：GREATER 优化流程
系统必须实现基于 "Gradient Over Reasoning" 的四阶段迭代优化流程，以利用小模型自身梯度优化 Skill 提示词。

#### 场景：端到端优化 (Refined)
当用户启动优化任务时，系统应按照以下逻辑顺序执行：
1. **生成推理 (Reasoning Generation)**: 基于当前 Control 生成推理链 $r$。
2. **构建上下文 (Context Construction)**: 将生成的推理链 $r$ 和提取提示词 $p_{extract}$ 插入到 Prompt 中。
3. **候选提案 (Candidate Proposal)**: 基于前缀 Logits 筛选候选词。
4. **梯度计算 (Gradient Computation)**: 基于完整的 `Prompt + Reasoning + Extract + Target` 上下文计算梯度。
5. **选择更新**: 选择最佳候选并更新。

### 需求：第二阶段 - 推理生成与答案提取 (Reasoning Generation)
系统必须强制模型生成推理过程，而非直接预测答案。
- **流程 (Update)**:
    1. **生成推理链**: 将 `[Prompt] + [Input]` 输入模型，采样生成推理文本 $r$ (Chain-of-Thought)。
    2. **动态构建**: 系统必须将生成的 $r$ 和配置的 `extract_prompt` (如 "Therefore, the final answer is") 拼接到上下文中，形成 `[Prompt] + [Input] + [Reasoning Chain] + [Extract Prompt]`。
    3. **目标对齐**: `Target` (Ground Truth Answer) 必须紧跟在 `Extract Prompt` 之后。

#### 场景：构建上下文 (Refined)
在计算梯度前，System 动态更新 `SkillPrompter`，使其包含当前轮次生成的推理内容和提取引导词，确保 Loss 能够反向传播经过推理过程。

### 需求：数据集驱动优化 (Dataset-Driven Optimization)
系统必须支持从外部数据源（如 JSONL 文件）加载训练样本用于优化过程。
- **Final Target 提取**: 针对 GSM8K，加载器必须能够从 Answer 字段中分离出 Reasoning (如有) 和 Final Answer。通常 GSM8K Answer 包含解释，最后以 `####` 分隔答案。系统应将 `####` 后的内容作为 `final_target` (用于 Focused Loss)，并将完整 Answer (或仅 Final Answer) 作为 `target`.
    - **规范**: 若 Target 仅设为 Final Answer，则模型被强制学习从 Extract Prompt 直接输出答案。

#### 场景：GSM8K 加载
加载器读取 "The answer is... #### 42"，设置 `target="42"`, `final_target="42"`. 优化过程中生成前面的推理。
