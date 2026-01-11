# Design: Dynamic Reasoning & Extract Prompt

## Architecture

### 1. SkillPrompter Upgrade
Add fields:
- `reasoning`: Optional[str] - The generated chain of thought.
- `extract_prompt`: Optional[str] - The bridge string (e.g. "Therefore...").

Behaviors:
- `_update_ids` must sequence: `[Header] [Goal] [Control] [AsstHeader] [Reasoning] [ExtractPrompt] [Target]`.
- `_loss_slice` must point to `[Target]`.
- `_assistant_role_slice` should probably cover up to `[ExtractPrompt]` end? Or just the header?
    - Usually `_assistant_role_slice` is used to define where "Response" starts for slicing.
    - We mainly care about `_control_slice` (for coordinate finding) and `_target_slice` (for loss).

### 2. GreaterOptimizer Flow
In `optimize` loop:
1. `greater_core.generate_reasoning(...)` -> returns `r`.
2. Update `prompter.reasoning = r`.
3. Update `prompter.extract_prompt = config.extract_prompt` (e.g. "Therefore the answer is").
4. `prompter._update_ids()` (Re-tokenize).
5. `greater_core.propose_candidates(...)` (Note: candidates are for *Control*. Control is before Reasoning. Does Reasoning downstream affect Control logits? No, Control is past. But we need gradients from Reasoning+Target back to Control.)
    - Wait. `propose_candidates` proposes *replacements* for Control. It uses the `input_ids`.
    - If `input_ids` now contains `Reasoning + Target`, `propose_candidates` (Forward Pass) uses the full context?
    - Reference implementation: `propose_candidates` typically uses a shorter context or just the prefix?
    - Actually candidates are proposed based on `Grad` (Hotflip) or just `Logits`?
    - `GreaTer` uses `propose_candidates: Union of Intersections` from *logits at Control positions*. This depends only on `Goal + Control_Prefix`. It does NOT depend on Reasoning (which is downstream).
    - So `SkillPrompter` update affects `compute_gradient` (backward pass), but strictly specific `propose_candidates` (if it just looks at control logits) might not need the full tail.
    - HOWEVER, `propose_candidates` in our code takes `input_ids_batch`. If we pass the full `SkillPrompter.input_ids`, it's fine. It just looks at `position_idx`.

### 3. Focused Loss
- In `_load_training_data`: Extract `final_target` strictly.
- For GSM8K: `Target` = "#### 42". `Final Target` = "42".
- `SkillPrompter`: Support `final_target` (already there).

## API Changes
- `SkillPrompter.__init__`: Add `extract_prompt`.
- `SkillPrompter.update_reasoning(text)`: New method.
