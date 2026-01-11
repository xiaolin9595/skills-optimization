# Task List

## Optimizer Updates
- [x] Update `GreaterOptimizer._load_training_data_raw` to support BBH CSV format. <!-- id: support-bbh-csv -->
    - Use `csv` module for robust parsing.
    - Fallback to JSONL for GSM8K compatibility.
- [x] Verify BBH data loading with a unit test. <!-- id: test-bbh-loading -->

## Experiment Setup
- [x] Create `experiments/run_bbh.py`. <!-- id: create-bbh-script -->
    - Set dataset path to `referenceSolution/GreaTer/data/BBH/boolean_expressions.json`.
    - Set initial prompt to the official BBH string.
    - Set extraction prompt for boolean expressions.
- [x] Run a minimal iteration of BBH optimization to verify the flow. <!-- id: verify-bbh-flow -->
