## 修改需求

### 需求：Conversation Template 支持
系统必须支持主流微调模型的对话模板（如 Llama-2, Llama-3 格式），以确保优化过程中的 Prompt 结构与模型预训练格式一致。
- **功能**:
    - 支持 `llama-2`, `llama-3` 等常见模板。
    - **Llama-3 支持**: 必须针对 Llama-3 的特殊标记（如 `<|start_header_id|>`, `<|eot_id|>`）实现精确的手动构建和 Slicing 逻辑。
    - 能够精确识别 Prompt 中的 User/Assistant 角色、Goal、Control、Target 等部分的 Token Slice 范围。
    - 支持在 Template 环境下动态更新 Control 部分的 Token。

#### 场景：Llama-3 适配
当配置 `template_name="llama-3"` 时，系统生成符合 Llama-3 Instruct 格式的 Prompt，并正确计算 Assistant 回复部分的 Loss Slice，避开 Header 标记。

## 新增需求

### 需求：数据集驱动优化 (Dataset-Driven Optimization)
系统必须支持从外部数据源（如 JSONL 文件）加载训练样本用于优化过程。
- **配置**: `OptimizeConfig` 需支持 `dataset_path` 和 `num_examples` 参数。
- **加载器**: 针对 GSM8K 等标准格式（`question`, `answer` 字段），实现自动解析和加载。

#### 场景：真实数据实验
用户指定 GSM8K 的 `train.jsonl` 路径，系统自动加载前 5 个样本的 Question 作为 Goal，Answer 作为 Target，进行 Prompt 优化。
