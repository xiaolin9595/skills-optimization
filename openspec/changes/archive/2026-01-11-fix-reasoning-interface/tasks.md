# Task List

## Core Updates
- [x] Refactor `greater_core.generate_reasoning` to new signature. <!-- id: refactor-core -->
    - remove `prompt_ids`, `extract_prompt_ids`.
    - support batching.
    - support `decode` and `return_only_new`.
- [x] Update `greater_core.select_and_update` to match new signature. <!-- id: update-caller-core -->
    - Concat prompt + input inputs.
    - Manually append `extract_prompt_ids` to the result if needed.
    - Call with `decode=False`, `return_only_new=False`.

## Optimizer Updates
- [x] Update `greater.py` loop call to `generate_reasoning`. <!-- id: update-caller-opt -->
    - Pass concatenated batch (already doing so).
    - Set `decode=True`.
    - Remove incompatible args (`extract_prompt_ids` removed).
