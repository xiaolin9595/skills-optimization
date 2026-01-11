# Fix GreaTer Implementation Gaps

## 为什么
当前的 GreaTer 实现与 Reference Solution 相比存在多个关键差距，导致算法的数学正确性和实际效果受到影响。主要问题包括梯度归一化缺失、Focused Loss 未实现、Non-ASCII 过滤缺失、梯度聚合方式错误以及缺少关键的 Sequential Increasing 和缓存机制。这些问题在 @[TerminalName: node, ProcessId: 3150] 中被明确指出。

## 变更内容
本变更将全面修复上述 12 个问题，使实现严格对齐官方 GreaTer 的逻辑。
1. **核心算法修正**:
   - 实现梯度 L2 归一化。
   - 修正梯度聚合逻辑（先归一化后累加）。
   - 实现 Non-ASCII Token 过滤。
2. **Loss 机制增强**:
   - 引入 `Focused Loss` 和 `Unfolding` 窗口机制，用于捕捉特定输出模式。
   - 修正 Llama-2 的 Loss Slice 计算。
   - 增加 Temperature 参数支持。
3. **优化策略增强**:
   - 实现 `Sequential Increasing` 策略（动态增加 Control 长度）。
   - 添加 `Loss Cache` 避免重复计算。
   - 添加 Early Stopping。
4. **基础设施增强**:
   - 完善 Tokenizer 的模型特定配置（Pad/BOS/EOS）。
   - 增强 `SkillPrompter` 支持 `final_target`。

本变更暂不引入多进程 Worker 机制（Issue 8），优先保证算法逻辑正确性，保持单进程实现。
