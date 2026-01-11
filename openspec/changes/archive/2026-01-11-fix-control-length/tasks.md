# Task List

## Logic Implementation
- [x] In `GreaterOptimizer.optimize`, implement initial length check and resizing logic before the main loop. <!-- id: impl-init-resize -->
    - Tokenize `skill.content`.
    - Truncate or Pad with `!` to match `config.start_len`.
    - Update `current_control_text`.
- [x] In the loop `if length_iter > config.start_len` block: <!-- id: impl-loop-resize -->
    - Ensure we append exactly one token.
    - Validate that `len(tokenizer(new_text))` equals `length_iter`.

## Verification
- [x] Verify using existing `run_gsm8k.py` setup (which has the mismatch scenario).
    - Logs should show "Truncated control..." or "Padded control...".
    - Actual execution should show `control_len` matching `length_iter`.
