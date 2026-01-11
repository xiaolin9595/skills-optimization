# Proposal: Validate GreaTer with Llama-3 and GSM8K

## Background
The current GreaTer implementation has been refined fixing algorithms and logic, but relies on dummy data and only supports Llama-2 templates. The user provides a Llama-3 model (`llama-3.2-3b-instruct`) and access to the GSM8K dataset from the reference project. To validate the reproduction physically, we need to adapt the system to this model and data.

## Goal
Execute a real-world optimization experiment using:
1.  Model: `/Volumes/TSU302/models/llama-3.2-3b-instruct`
2.  Dataset: `referenceSolution/GreaTer/data/grade_school_math/data/train.jsonl`
3.  Target: Optimize a math solving prompt/skill.

## Scope
1.  **Llama-3 Support**: Extend `SkillPrompter` and `ConversationTemplate` to strictly support Llama-3 chat formatting and slicing.
2.  **Dataset Loading**: Implement `GSM8K` data loading logic within `GreaterOptimizer` or via `OptimizeConfig`.
3.  **Experiment Orchestration**: Create a script to setup the environment and run the optimization Loop.

## Risks
- **Llama-3 Slicing**: Llama-3's tokenizer behavior (special tokens like `<|eot_id|>`) is more complex than Llama-2. Incorrect slicing will break Gradient computation. We must verify slicing logic carefully using token ID inspection.
- **Memory**: 3B model should fit on most GPU/MPS, but gradients + optimizer state might push limits on smaller devices. We'll stick to `batch_size=1` for safety.
