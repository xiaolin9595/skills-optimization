## 修改需求

### 需求：动态控制长度与搜索策略 (Dynamic Control Strategy)
系统必须支持动态调整 Control Prompt 的长度，以平衡搜索空间和表达能力。

#### 场景：长度初始化与强制对齐 (New)
在开始优化前，系统必须将初始 Skill Content 的长度调整为配置的 `start_len`。
- **截断**: 若输入 Content 长度超过 `start_len`，系统必须将其截断，保留前 `start_len` 个 Token。
- **填充**: 若输入 Content 长度不足，系统必须使用占位符（如 `!`）填充至 `start_len`。
- **目的**: 确保优化过程严格遵循 `start_len` 到 `end_len` 的维度递增逻辑，避免因初始 Prompt 过长导致的维度错配和资源浪费。

#### 场景：递增增长
当 `length_iter` 增加时，系统必须在当前最优 Control 后追加**正好一个**占位符 Token，确保搜索空间维度精确加一。
