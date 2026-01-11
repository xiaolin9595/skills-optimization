# 任务列表

## Phase 1: Infrastructure & Prompter
- [x] 引入 `fastchat` 依赖 (或简单的 Template 适配层) 并实现 `src/skill_opt/optimizer/prompter.py` <!-- id: impl-prompter -->
    -   需支持 Llama-2/3 等常见 Template。
    -   实现 `_update_ids` 和 `slice` 计算逻辑 (Goal/Control/Target/Loss Slices)。
- [x] 更新 `utils.py` 支持加载 `Prompter` 所需的配置。 <!-- id: update-utils -->

## Phase 2: Core Algorithm Refinement
- [x] 重构 `greater_core.py:propose_candidates` <!-- id: refine-candidate -->
    -   实现 Reference 中的 "Union of Intersections" 逻辑。
    -   支持 Batch Logits 输入。
- [x] 重构 `greater_core.py:compute_gradient` <!-- id: refine-gradient -->
    -   支持多 Loss (Target Loss + Control Loss)。
    -   使用精确的 Slice 进行梯度回传。

## Phase 3: Optimizer Integration
- [x] 更新 `GreaterOptimizer.optimize` 主循环 <!-- id: update-optimizer -->
    -   使用 `SkillPrompter` 管理 Prompt 构建。
    -   处理 Conversation Template 带来的 Input 长度变化。
- [x] 验证测试 `tests/test_refined_greater.py` <!-- id: verify-refinement -->
