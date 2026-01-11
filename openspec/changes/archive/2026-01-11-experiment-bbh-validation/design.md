# Design: BBH Validation

## Data Loading Logic
The BBH files in the reference solution are formatted as CSV but named with `.json`.
They contain columns: `goal`, `target`, `final_target`.
`GreaterOptimizer._load_training_data_raw` will be updated to:
1.  Read the first line of the file.
2.  If the line starts with `goal,target,final_target`, use `csv.DictReader` to parse the file.
3.  Otherwise, fallback to the existing JSONL loading logic.

## Initial Prompt & Extractor
The official BBH initial prompt for GreaTer is:
`" proper logical reasoning and think step by step. Finally give the actual correct answer."`

The extractor for `boolean_expressions` is:
`"Therefore, the final answer (use exact format: '$ True' or '$ False') is $ "`

Note: The `$` in the extractor is used as a separator/marker in GreaTer's code. In our implementation, `SkillPrompter` appends the `extract_prompt` text. We should ensure the `final_target` extraction logic in `SkillPrompter` (which uses Focused Loss) is compatible.
Currently, `SkillPrompter` constructs:
`[User+Goal+Control] + [Assistant Header] + [Reasoning] + [Extract Prompt] + [Target]`
Loss is calculated on `[Target]`. 

## Experiment Orchestration
A new script `experiments/run_bbh.py` will be created, following the pattern of `run_gsm8k.py`.
It will target the `boolean_expressions.json` task as a representative BBH test.
It will use `Llama-3-8B-Instruct` as the base model.
