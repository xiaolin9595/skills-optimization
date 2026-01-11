# 任务列表

## Phase 1: 核心机制实现 (GreaTer 4-Stage Workflow)
- [x] 创建 `src/skill_opt/optimizer/greater_core.py`，实现 Stage 1 **候选提案逻辑** (Logits 计算 + 交集聚合) <!-- id: optim-stage-1 -->
- [x] 在 `greater_core.py` 中实现 Stage 2 **推理生成逻辑** (通过 CoT 生成推理链上下文) <!-- id: optim-stage-2 -->
- [x] 在 `greater_core.py` 中实现 Stage 3 **梯度计算逻辑** (稀疏嵌入梯度回传) <!-- id: optim-stage-3 -->
- [x] 在 `greater_core.py` 中实现 Stage 4 **选择与更新逻辑** (Forward 验证 + 择优) <!-- id: optim-stage-4 -->
- [x] 编写单元测试 `tests/test_greater_core.py` 验证以上各阶段函数的正确性 <!-- id: test-core-logic -->

## Phase 2: 优化器类封装
- [x] 创建 `src/skill_opt/optimizer/greater.py` 实现 `GreaterOptimizer` 类，集成 `greater_core` 逻辑 <!-- id: create-class -->
- [x] 实现模型加载工具 `src/skill_opt/optimizer/utils.py` (支持 Config 读取与 Device 分配) <!-- id: impl-utils -->
- [x] 实现 `optimize` 主循环：遍历位置 -> 调用 4 阶段流程 -> 迭代更新 <!-- id: impl-loop -->
- [x] 编写集成测试 `tests/test_optimizer.py` 使用 Mock 模型验证完整优化流程 <!-- id: test-optimizer -->

## Phase 3: 系统集成
- [x] 更新 `AppConfig` 添加 GreaTer 专属配置 (如 `top_k`, `top_mu`, `max_steps`) <!-- id: update-config -->
- [x] 在 `src/skill_opt/optimizer/__init__.py` 导出 `GreaterOptimizer` <!-- id: export -->
- [x] 验证全流程：使用 `OptimizeConfig` 运行一次极简优化任务 <!-- id: verify-full -->
