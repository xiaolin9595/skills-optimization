# Design: Refined GreaTer Optimizer

## Context
Alignment with GreaTer reference implementation to ensure effectiveness on Chat Models.

## Core Concepts
1.  **Prompter Management**: Encapsulate Template + Tokenization + Slicing logic.
2.  **Strict GreaTer Workflow**:
    -   **Intersection**: Union of intersections of random subsets of Top-K logits.
    -   **Slices**: `_control_slice`, `_target_slice`, `_loss_slice` dynamically updated.
    -   **Loss**: `Control Loss` (divergence from repetitive patterns if any, or just keeping control stable) + `Target Loss` (Main Objective).

## Component Redesign

### 1. `SkillPrompter` (New Class)
Adapts `referenceSolution/GreaTer/llm_opt/base/attack_manager.py:Prompter`.
-   **Fields**: `tokenizer`, `conv_template`, `goal`, `target`, `control`.
-   **Methods**: `get_prompt()`, `update_control()`, `get_slices()`.
-   **Responsibility**: Converting abstract (Skill, Task) -> Model Input IDs with correct Chat Format and identifying Slices.

### 2. `greater_core.py` (Refinement)
-   `propose_candidates`:
    -   Input: `SkillPrompter` (or sliced logits context), `batch_logits`.
    -   Logic:
        -   Compute Logits for Control positions.
        -   Loop Logic: Random subsampling intersection (Ref `morph_control`).
-   `compute_gradient`:
    -   Input: `complete_input_ids`, `slices` (`control`, `target`, `loss`).
    -   Logic:
        -   Sparse Embeddings on `control_slice`.
        -   Optimization Target: Minimize loss on `loss_slice` (Target / Focused).
        -   Maybe `control_loss` if needed (Reference has `control_weight * control_loss`).
-   `select_and_update`:
    -   Logic: Validate Top-Mu.

### 3. Loop & State
`GreaterOptimizer` maintains the `SkillPrompter` instance.
State: `current_control_ids`.

## Data Flow
Loop:
  1. `Prompter` -> `input_ids` (with template) -> `slices`.
  2. `propose_candidates(input_ids, control_slice)` -> `candidates`.
  3. `compute_gradient(input_ids, control_slice, loss_slice, candidates)` -> `gradients`.
  4. `select_and_update` -> `best_new_control`.
  5. Update `Prompter.control`.
