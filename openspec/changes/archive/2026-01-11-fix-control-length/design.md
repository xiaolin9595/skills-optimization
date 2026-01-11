# Design: Enforced Control Length

## Logic Flow

### 1. Initialization Phase (`GreaterOptimizer.optimize`)
Before entering the `length_iter` loop:
1.  **Tokenize** `skill.content` to get initial IDs.
2.  **Compare** `len(ids)` with `config.start_len`.
3.  **Resize**:
    - If `len > start_len`: `ids = ids[:start_len]` (Truncate).
    - If `len < start_len`: Append `!` token IDs until `len == start_len`.
4.  **Decode** back to string `current_control_text`.

### 2. Iteration Phase
Inside `for length_iter in range(start_len, end_len + 1)`:
1.  **Verify/Enforce Again**:
    - Ideally step 1 sets us up for `length_iter == start_len`.
    - If `length_iter > start_len` (i.e. we are growing):
        - We take the *optimized* control from previous step.
        - We append ONE token `!` to it.
        - Verify new length is exactly `length_iter`. (Re-tokenize to be sure, handle potential token merging weirdness although `!` is usually stable).

## Component Changes

### `GreaterOptimizer`
- Modify `optimize` method.
- Add helper or inline logic for "Resize to N tokens".

### `AppConfig` / `OptimizeConfig`
- No changes needed, we just enforce the existing parameters.

## Algorithm Alignment
This aligns with GreaTer's "Coordinate Gradient" approach where the search space dimensionalty is fixed to `N` at step `N`.
