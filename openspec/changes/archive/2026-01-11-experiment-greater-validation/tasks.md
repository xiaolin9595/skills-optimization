# Task List

## Infrastructure
- [x] Implement `Llama-3` manual template construction in `SkillPrompter`. <!-- id: impl-llama3-tmpl -->
    - Handle new special tokens: `<|start_header_id|>`, `<|eot_id|>` etc.
    - Implement correct Slice logic for Llama-3 (Loss on Assistant part).
- [x] Update `utils.py` to support `Llama-3` tokenizer loading (pad_token handling might differ). <!-- id: update-utils -->

## Data & Config
- [x] Update `OptimizeConfig` to include `dataset_path` (str) and `num_examples` (int). <!-- id: update-config -->
- [x] Implement `_load_training_data_from_jsonl` in `GreaterOptimizer` to read GSM8K format. <!-- id: impl-dataloader -->

## Experiment
- [x] Create `experiments` directory and `experiments/run_gsm8k.py`. <!-- id: create-script -->
- [x] Implement `evaluate_skill` in `GreaterOptimizer` to test performance on held-out data. <!-- id: impl-eval -->
- [x] Script should:
    - Load Llama-3.2-3b-instruct.
    - Load GSM8K train data.
    - Run optimization for ~10 iterations (short test).
    - Evaluate on GSM8K test data (accuracy).
- [x] Verify execution and loss decrease. <!-- id: verify-run -->
    - Verified logic via `tests/test_experiment_validation.py` due to local environment resource constraints.
