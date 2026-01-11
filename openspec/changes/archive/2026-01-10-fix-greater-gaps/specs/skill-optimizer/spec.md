## 修改需求

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

## 新增需求

### 需求：动态控制长度与搜索策略 (Dynamic Control Strategy)
系统必须支持动态调整 Control Prompt 的长度，以平衡搜索空间和表达能力。
- **Sequential Increasing**: 初始使用较短的 Control Length (`start_len`)，当 Loss 在 `patience` 步数内未下降（Plateau）时，系统必须自动增加 Control 长度（追加 Token），直到达到 `end_len`。
- **Loss Cache**: 在候选词选择（Selection）阶段，系统必须缓存 `(prompt_ids) -> loss` 的映射，避免对同一 Prompt 重复执行昂贵的 Forward Pass 验证。
- **Early Stopping**: 当 Loss 低于阈值或长时间未下降时提前终止优化。

#### 场景：自适应优化
当初始长度为 3 的 Control 无法进一步降低 Loss 时，系统自动将其扩展为 4 个 Token，从而解锁更大的优化空间，同时利用缓存加速此过程。

### 需求：候选词过滤 (Token Filtering)
系统必须确保生成的 Control Prompt 具有可读性并且不包含特殊字符。
- **Non-ASCII 过滤**: 在 Stage 1 (Candidate Proposal) 计算 Logits 时，系统必须将 Non-ASCII Token 和不可打印字符的概率设为 `-inf`，仅允许 ASCII 字符候选。

#### 场景：合法性约束
在生成候选词时，系统自动屏蔽乱码或不可见字符，确保优化得到的 Prompt 是人类可读的英文文本。

### 需求：模型适配增强
系统必须针对不同模型架构（如 Llama-2, Llama-3, Falcon）正确配置 Tokenizer 的 Padding Side, Pad Token, BOS/EOS Token，并必须使用正确的 Slice Offset（如 Llama-2 的 `-3` Loss Slice 修正）以对齐模型内部机制。

#### 场景：模型兼容性
当切换到 Llama-2 模型时，系统自动识别并应用 `-3` 的 Loss Slice 偏移量，以规避 EOS Token 对梯度的干扰。
