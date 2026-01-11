# 提案：实现 GreaTer 技能优化器

## 背景
根据系统核心架构的设计，我们需要实现 `SkillOptimizer` 接口的具体逻辑。GreaTer (Gradient Over Reasoning) 是本项目的核心算法，利用开源小模型的梯度信息来优化 Skill 提示词。

## 目标
移植并适配 GreaTer 算法核心逻辑，实现 `GreaterOptimizer` 类。

## 范围
- **包含**:
    - 移植 `referenceSolution/GreaTer/llm_opt/minimal_gcg/opt_utils.py` 中的核心梯度计算与采样逻辑。
    - 实现 `src/skill_opt/optimizer/greater.py`。
    - 实现 `src/skill_opt/optimizer/utils.py` (模型加载、Token 处理工具)。
    - 集成 `OptimizeConfig` 以控制超参数（学习率、迭代次数等）。
- **不包含**:
    - PromptBridge 转移逻辑。
    - iFlow 执行器集成。
    - 大规模并发优化支持 (本次仅实现单卡/单批次逻辑)。

## 成功标准
- `GreaterOptimizer` 能够被实例化并调用 `optimize` 方法。
- 能够在一个 Mock 或真实小模型（如 GPT-2/TinyLlama，用于测试）上运行完整的梯度下降循环。
- `optimize` 方法返回更新后的 `OptimizedSkill`。
