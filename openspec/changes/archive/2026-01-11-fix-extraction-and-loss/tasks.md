# Task List

## Infrastructure
- [x] Update `SkillPrompter` to support `reasoning` and `extract_prompt` fields. <!-- id: impl-prompter-extract -->
- [x] Update `SkillPrompter._update_ids` to construct `... [Reasoning] [ExtractPrompt] [Target]`. <!-- id: impl-prompter-logic -->
- [x] Add `update_reasoning` method to `SkillPrompter`. <!-- id: impl-prompter-update -->

## Optimizer Integration
- [x] Update `GreaterOptimizer._load_training_data_raw` (or `_from_jsonl`) to extraction `final_target` for GSM8K (split by "####"). <!-- id: impl-data-extract -->
- [x] In `GreaterOptimizer.optimize` loop:
    - [x] Perform `generate_reasoning` (already exists, verify integration).
    - [x] Update `prompters` with generated reasoning and extraction prompt.
    - [x] Ensure `compute_gradient` uses the updated prompters (which now include reasoning path). <!-- id: impl-opt-loop -->

## Config
- [x] Add `extract_prompt` to `OptimizeConfig` (default: "Therefore, the answer is"). <!-- id: impl-config -->
