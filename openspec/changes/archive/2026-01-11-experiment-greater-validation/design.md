# Design: Llama-3 & GSM8K Support

## Architecture Changes

### 1. SkillPrompter & Llama-3
Llama-3 uses a specific chat template:
```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{Goal} {Control}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{Target}
```
We need to map GreaTer's Goal/Control/Target to this structure.
- **Goal + Control** -> User Message
- **Target** -> Assistant Message (Teacher Forcing)

**Slicing Strategy**:
Instead of heuristic strings search (which fails with special tokens), we will use `tokenizer.apply_chat_template` to generate the base structure, then identify the token indices of `Goal`, `Control` and `Target` by looking for their tokenized sequences.
- *Correction*: `apply_chat_template` outputs a string or IDs. If we use IDs, finding the sub-sequence of `Control` (which might be tokenized differently due to context) is risky.
- *Refined Strategy*: Construct the prompt *manually* using the known Llama-3 special tokens (`<|start_header_id|>`, etc.) to ensure we know exactly where each part starts and ends. This aligns with how `SkillPrompter` handles `llama-2` manually.

### 2. Data Loading
The `GreaterOptimizer.optimize` currently has hardcoded data. We will enhance `OptimizeConfig` to accept `dataset_path`.
- We will add a `GSM8KLoader` utility or method in `greater.py`.
- It will parse `.jsonl` files (Key fields: `question`, `answer`).
- It will extract a subset (e.g., first 4 examples) for the experiment to save time.

### 3. Usage
User will run:
```bash
python experiments/run_gsm8k_optimization.py
```
This script will:
1. Initialize `AppConfig` with Llama-3 path.
2. Initialize `OptimizeConfig` with data path.
3. Call `optimizer.optimize`.
