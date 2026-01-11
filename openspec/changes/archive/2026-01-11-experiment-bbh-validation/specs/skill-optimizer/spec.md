## 修改需求

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
