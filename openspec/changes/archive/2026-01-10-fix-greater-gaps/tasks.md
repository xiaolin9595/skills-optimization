# 任务列表

## Phase 1: Core Math & Loss Fixes
- [x] 修正 `greater_core.py:compute_gradient` 返回梯度前进行 L2 归一化 (或者确认由调用方处理，保持接口一致性)。 <!-- id: fix-grad-norm -->
    - *Decision*: Reference normalize in `attack_manager.py` (caller). `greater_core` as a library function might return raw. 但是为了安全，我们可以在 core 中提供选项或者在 Optimizer 中修正。依照 Issue 1 描述，建议在 Optimizer 聚合时处理，或者 Core 明确返回 Normalized。此处遵循 Reference: Core calculates, Caller normalizes.
- [x] 在 `GreaterOptimizer` 中修正梯度聚合逻辑：先 Normalize 再 Accumulate。 <!-- id: fix-grad-agg -->
- [x] 在 `greater_core.py` 中实现 `Focused Loss` 计算逻辑 (Unfolding windows)。 <!-- id: impl-focused-loss -->
    - 需引入 `focused_target` 参数。
- [x] 修正 `SkillPrompter` 的 Slice 计算逻辑，特别是 Llama-2 (`-3` offset) 和 `_focused_target_slice`。 <!-- id: fix-slices -->

## Phase 2: Filtering & Constraints
- [x] 在 `greater_core.py:propose_candidates` 中实现 `Non-ASCII` 过滤。 <!-- id: impl-ascii-filter -->
    - 需要在 `utils.py` 或 `greater_core.py` 中添加 `get_nonascii_toks` 辅助函数。
- [x] 在 `config.py` 和 `greater_core.py` 中集成 `temperature` 参数。 <!-- id: add-temperature -->

## Phase 3: Optimization Loop Enhancements
- [x] 在 `GreaterOptimizer` 中实现 `Loss Cache`。 <!-- id: impl-cache -->
- [x] 在 `GreaterOptimizer` 中实现 `Sequential Increasing` 控制长度策略。 <!-- id: impl-seq-inc -->
    - 更新 `OptimizeConfig` 添加 `start_len`, `end_len`, `patience`。
- [x] 在 `GreaterOptimizer` 中实现 `Early Stopping`。 <!-- id: impl-early-stop -->
- [x] 完善 `utils.py` 中的模型特定 Tokenizer 配置 (Pad/BOS/EOS)。 <!-- id: fix-tokenizer -->

## Phase 4: Validation
- [x] 编写测试 `tests/test_greater_fixes.py` 验证 Focused Loss, Non-ASCII Filter 和 Cache 机制。 <!-- id: verify-fixes -->
