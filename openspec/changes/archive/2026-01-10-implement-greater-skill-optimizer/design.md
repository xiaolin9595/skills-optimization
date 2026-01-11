# 设计：GreaTer 优化器实现

## 核心组件

### 1. `GreaterOptimizer` 类 (Top-Level Wrapper)
继承自 `SkillOptimizer`，负责编排优化流程。

**属性**:
- `model`: `AutoModelForCausalLM`
- `tokenizer`: `AutoTokenizer`
- `device`: `torch.device`

**方法**:
- `__init__(self, config: AppConfig)`: 加载模型和工具。
- `optimize(self, skill: Skill, config: OptimizeConfig) -> OptimizedSkill`: 主循环控制。

### 2. `greater_core` 模块 (Algorithm Logic)
实现具体的 GreaTer 算法四阶段核心函数。

**主要函数**:
- `propose_candidates(model, prompt_tokens, batch_inputs) -> SparseEmbeddings`: **Stage 1** 基于 logits 交集筛选候选。
- `generate_reasoning(model, prompt, input) -> Context`: **Stage 2** 生成推理链并构建完整上下文。
- `compute_gradient(model, context, candidates) -> Gradients`: **Stage 3** 基于推理链的稀疏梯度反向传播。
- `select_and_update(model, prompt, gradients, candidates) -> NewPrompt`: **Stage 4** 基于梯度排序和前向验证择优更新。

## 算法流程 (Optimize Method)

1.  **初始化**: 构建初始 Skill Prompt，加载数据集 Batch。
2.  **迭代**: 对 Skill 中的每个 Token 位置 `i` 进行循环优化：
    a.  **Stage 1: 提案**: 计算当前位置的 Logits，取 Batch 交集，构建稀疏候选嵌入。
    b.  **Stage 2: 推理**: 对 Batch 样本生成 "Skill + Input -> CoT -> Answer" 的完整上下文。
    c.  **Stage 3: 梯度**: 反向传播计算 Loss 对 Stage 1 候选嵌入的梯度。
    d.  **Stage 4: 这里**: 选取负梯度最大的 Top-$\mu$ 候选，结合实测 Loss 决定是否更新位置 `i` 的 Token。
3.  **循环**: 重复上述过程直至达到 `Max Steps`。
4.  **输出**: 返回优化后的 Skill。

## 代码结构

```python
# src/skill_opt/optimizer/greater.py
class GreaterOptimizer(SkillOptimizer):
    def optimize(self, skill, config):
        # 负责迭代循环和调度 greater_core 中的函数
        pass

# src/skill_opt/optimizer/greater_core.py
def propose_candidates(...): pass
def generate_reasoning(...): pass
def compute_gradient(...): pass
def select_and_update(...): pass
```

## 依赖
- `torch`: 梯度计算与张量操作。
- `transformers`: 模型推理与 Tokenizer。

