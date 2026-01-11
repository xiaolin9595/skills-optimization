# Proposal: Fix Reasoning Generation Interface

## Problem
The optimization loop implemented in `greater.py` calls `greater_core.generate_reasoning` with a batch of input IDs (`input_ids_batch`), expecting a list of decoded strings. However, `greater_core.generate_reasoning` currently:
1.  Expects separate `prompt_ids` and `input_ids`.
2.  Does not handle batch inputs correctly (unconditionally unsqueezes, assuming single sample).
3.  Returns a `torch.Tensor` (IDs) instead of decoded strings, while `greater.py` expects strings to pass to `SkillPrompter`.
4.  Is also used by `select_and_update` in `greater_core.py`, which *expects* it to return IDs and takes separate inputs.

## Solution
Refactor `greater_core.generate_reasoning` to be a robust, batch-aware utility that can return either IDs or Text.

1.  **Unified Signature**:
    ```python
    def generate_reasoning(
        model, 
        tokenizer,
        input_ids: torch.Tensor, # Batch or Single [B, Seq] or [Seq]
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        decode: bool = True # If True, returns List[str]. If False, returns Tensor.
    ) -> Union[List[str], torch.Tensor]:
    ```

2.  **Implementation Updates**:
    - Remove `prompt_ids` argument. Caller must concatenate `prompt + input`.
    - Handle 1D vs 2D input (if 1D, unsqueeze to 2D for `generate`, then squeeze back if needed).
    - If `decode=True`: Use `tokenizer.batch_decode(generated_ids, skip_special_tokens=True)`.
    - If `decode=False`: Return `generated_ids`. Note: Model generate typically returns `[Context + Generated]`. We should return *only* the `Generated` part (or `Context + Generated` based on usage).
      - `greater.py` needs **Reasoning Text** (generated part only).
      - `select_and_update` needs **Full Sequence** (Context + Reasoning) to cat with Target?
      - Wait. `select_and_update` logic: `full_seq = generate...`. Then `input_w_gt = cat([full_seq, val_lbl])`.
      - It expects `full_seq` to be [Prompt + Input + Reasoning].
      - `greater.py` needs just [Reasoning] string.
      - **Decision**: `generate_reasoning` should return **Full Sequence IDs** if `decode=False` (standard `generate` behavior), but if `decode=True`, it takes the *generated part only* to decode (more useful for injecting into prompts).
      - *Refinement*: Let's stick to standard behavior. `decode=True` returns decoded text of the *generated* part. `decode=False` returns full IDs.
      - Wait, if `decode=False` returns full IDs, `greater.py` could decode manually. But `greater_core` utils should handle decoding to be user-friendly.

    - **Update Callers**:
        - `greater.py`: Pass `input_ids_batch`. set `decode=True`.
        - `greater_core.py` (`select_and_update`): Concatenate `prompt + input` before calling. Set `decode=False`.

## Risks
- Breaking `select_and_update` if not careful with shapes.

## Delimitations
- Only changes `greater_core.py` interfaces and `greater.py` calls.
