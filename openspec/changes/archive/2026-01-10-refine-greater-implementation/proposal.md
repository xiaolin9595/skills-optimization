# Refine GreaTer Implementation

## 为什么
当前 GreaTer 优化器实现虽然完成了基础框架，但在核心机制上与 Reference Solution (@referenceSolution/GreaTer) 存在显著差距，导致其无法在真实模型上达到预期效果。主要差距包括：
1.  **缺少 Conversation Template 支持**：未适配 Chat 模型（如 Llama-2/3）的对话模板，导致 Prompt 结构错误。
2.  **Slice 机制不完整**：缺乏精细的 Slice 管理（Goal, Control, Target, Loss, Focused），导致梯度计算和 Loss 验证目标不精确。
3.  **候选生成策略偏差**：仅实现了简单的集合交集，未实现 Reference 中的"Logits Intersection via Union of Subsets"机制，可能导致候选集过小或为空。
4.  **Loss 函数简化**：缺少 `control_loss` 和 `focused_loss`，优化目标单一。
5.  **缺少关键优化**：未实现 `sequentially_increasing` 控制长度策略和 `loss_cache`。

## 变更内容
本变更将对 `greater_core.py` 和 `GreaterOptimizer` 进行重构，以严格对齐 Reference Solution 的核心逻辑：
1.  引入 `fastchat` 或类似机制支持 Conversation Templates。
2.  重构 `propose_candidates` 实现 Reference 的 Intersection 逻辑。
3.  重构 `compute_gradient` 和 `select_and_update` 支持 Slice 和多重 Loss。
4.  引入 `Prompter` 类或类似逻辑管理 Prompt 结构和 Slices。
