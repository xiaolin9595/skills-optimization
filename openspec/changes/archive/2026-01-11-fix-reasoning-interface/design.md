# Design: Reasoning Interface

## `generate_reasoning` Refactor

### Signature
```python
def generate_reasoning(
    model, 
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    decode: bool = True,
    return_only_new: bool = True # New flag to control if we return full seq or just new
) -> Union[List[str], torch.Tensor]:
```

### Logic
1.  **Input Normalization**:
    - If `input_ids` 1D -> Unsqueeze(0).
    - `input_len` = `input_ids.shape[1]`.
2.  **Generation**:
    - `model.generate(input_ids, ...)` -> `outputs` [B, Seq + New].
3.  **Processing**:
    - `new_tokens` = `outputs[:, input_len:]`.
    - If `decode`:
        - Return `tokenizer.batch_decode(new_tokens, skip_special_tokens=True)`.
    - Else:
        - If `return_only_new`: return `new_tokens`.
        - Else: return `outputs`.

## Call Site 1: `greater.py` (Optimize Loop)
- Needs: Generated strings (Reasoning only).
- Arguments: `input_ids=gen_batch`, `decode=True`.
- Returns: `List[str]`. Perfect.

## Call Site 2: `greater_core.py` (Selection)
- Needs: Full Sequence IDs (Prompt + Reasoning) to calculate Loss.
- Current Logic: `full_seq = generate...`. `cat([full_seq, target])`.
- Arguments: `input_ids=cat(prompt, val_in)`, `decode=False`, `return_only_new=False`.
- Returns: `Tensor` [1, Seq]. Perfect.

## API Changes
- `generate_reasoning`: Remove `prompt_ids`, `extract_prompt_ids`.
   - Caller handles `extract_prompt_ids` appending?
   - `extract_prompt` was in original `generate_reasoning` to append *after* generation?
   - Original `generate_reasoning` code:
     ```python
     full_sequence = outputs[0]
     if extract_prompt_ids is not None:
         full_sequence = torch.cat([full_sequence, extract_prompt_ids])
     return full_sequence
     ```
   - If we want to support this, we should add `append_ids`.
   - OR, caller does it.
   - `select_and_update` used `extract_prompt_ids`. It should append manualy.
   - `greater.py` handles extract prompt via `SkillPrompter` text usage.
   - Conclusion: Remove `extract_prompt_ids` from `generate_reasoning`. Caller handles logic.
