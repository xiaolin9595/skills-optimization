# Proposal: Fix Extract Prompt and Focused Loss

## Problem
1. **Extract Prompt Missing**: The current implementation does not support an "Extract Prompt" (e.g., "Therefore, the final answer is") inserted between the reasoning chain and the final answer. This is crucial for guiding the model to output the answer in the correct format and for aligning the loss calculation with the GreaTer methodology.
2. **Focused Loss Weakness**: The `final_target` used for focused loss calculation is currently just the full target string or manually passed. It needs to be more robustly extracted (e.g., just the numerical value "42" from "The answer is 42") to ensure the optimizer focuses on the *correctness* of the result, not just the reasoning verbiage.

## Solution
1. **Enhance SkillPrompter**: Add `extract_prompt` argument. Update `_update_ids` to insert `extract_prompt` after the Assistant Header (or reasoning placeholder) and before the `Target`.
    - Llama-2: `[INST] ... [/INST] {extract_prompt} {target}` (Wait, usually Extract Prompt is part of Assistant generation).
    - GreaTer Logic: `Input -> [Generate Reasoning] -> Append [Extract Prompt] -> Generate [Answer]`.
    - In **Optimization Mode** (Teacher Forcing): The `Target` string passed to Prompter *already contains* the full desired output (Reasoning + Answer)?
    - *Correction*: In GreaTer, the `Target` usually refers to the *Final Answer* we want to force. The `Reasoning` is generated dynamically during inference, but during *Optimization*, do we optimize against a ground truth reasoning?
    - Actually, GreaTer (Gradient Over Reasoning) implies we likely don't have ground truth reasoning (that's why we generate it).
    - But `SkillPrompter` is used for Gradient Calculation. Gradient is calculated on `Loss Slice`.
    - If we don't have ground truth reasoning, what is in `Target`?
    - If `Target` = "Answer".
    - Then `Prompt` = `Goal + Control`.
    - We forward pass.
    - If we enforce "Reasoning", we need `Target` to be "Reasoning + Answer"?
    - **Re-reading GreaTer paper/code intuition**: "Gradient over Reasoning" means we assume the model *generates* a reasoning path $r$. We want to optimize control $u$ such that $P(y|u, x, r)$ is maximized.
    - Implementation detail: We *sample* $r$ from model. Then we construct input `u + x + r`. Then we calculate gradient of $-\log P(y | u, x, r, p_{extract})$.
    - **Current Implementation Gap**: Our `GreaterOptimizer.optimize` loop generates `candidates`. But `compute_gradient` takes `context_ids_with_gt`. `context_ids_with_gt` comes from `prompters`. `prompters` are initialized with `target="Answer 1"`.
    - So `context_ids` = `User...Goal...Control...Asst...Answer`.
    - **Where is the reasoning $r$?**
    - The current implementation *skips* the generated reasoning $r$ in the gradient interface! `SkillPrompter` just uses the static `Target` (Ground Truth Answer).
    - **Critical Fix**: We need to *update* the `SkillPrompter` with the *generated reasoning* $r$ dynamically inside the optimization loop, before computing gradient.
    - And we need to insert `extract_prompt` between $r$ and `target`.

2. **Refine Logic**:
    - `GreaterOptimizer`: In `optimize` loop (specifically `compute_gradient` call preparation):
        - Generate Reasoning $r$ (already done in `generate_reasoning` in core, but need to be used).
        - Update `SkillPrompter` content to include $r$.
        - `SkillPrompter`: Support `reasoning` field.
        - `SkillPrompter`: Support `extract_prompt` field.
    - `SkillPrompter._update_ids`: Construct: `User...Goal...Control...Asst... {Reasoning} {ExtractPrompt} {Target}`.
    - `Loss Slice`: Should cover `Target`.

3. **Focused Target**:
    - Ensure `final_target` passed to `SkillPrompter` is the minimal concise answer.
    - Implement simple heuristic extraction (e.g. "####" splitter) if loading from GSM8K.

## Risks
- **Complexity**: Dynamically updating `SkillPrompter` structure implies `input_ids` size changes every step (or every epoch if we sample reasoning once).
- **Memory**: Longer context with reasoning.

## Delimitations
- We will implement this dynamics.
