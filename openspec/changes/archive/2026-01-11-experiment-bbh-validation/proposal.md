# Proposal: BBH Dataset Validation

## Background
We have successfully validated the GreaTer optimization framework using GSM8K. Now, we aim to extend this validation to the Big Bench Hard (BBH) dataset, which is the primary benchmark used in the original GreaTer paper. This requires supporting the CSV-based format used in BBH "JSON" files and using the official GreaTer initial prompts and extractors.

## Goals
1.  Update `GreaterOptimizer` to robustly load BBH data files (CSV format).
2.  Implement an experiment script for BBH tasks.
3.  Use the official BBH initial prompt for optimization.

## Proposed Changes

### `GreaterOptimizer` (src/skill_opt/optimizer/greater.py)
- Update `_load_training_data_raw` to:
    - Detect the file format.
    - If the first line is `goal,target,final_target`, use a CSV reader to parse the file.
    - Support quoted fields and multiline strings (common in BBH).

### Experiment Script (experiments/run_bbh.py)
- Create a new script `run_bbh.py`.
- Configure it to use `boolean_expressions.json` from the BBH dataset.
- Set the initial skill content to: `" proper logical reasoning and think step by step. Finally give the actual correct answer."`
- Set the `extract_prompt` to: `"Therefore, the final answer (use exact format: '$ True' or '$ False') is $ "`.

## Risks
- CSV parsing might be slower than JSONL for very large files, but BBH files are manageable.
- Llama-3 tokenization of the initial prompt must be verified to match expected `start_len`.
