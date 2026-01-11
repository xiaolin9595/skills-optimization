# Design: GreaTer Implementation Fixes

## Context
Fixing critical mathematical and logical discrepancies between current implementation and GreaTer reference.

## 1. Focused Loss & Window Unfolding
**Problem**: Standard CrossEntropyLoss on the whole target is insufficient for "Focused" attacks where we care about specific keywords appearing *anywhere* or in a specific relation.
**Solution**:
- **Input**: `focused_target` (string/tokens).
- **Mechanism**:
    1. Identify `loss_logits` corresponding to the generated text.
    2. Use `torch.Tensor.unfold` to create sliding windows of size `len(focused_target)` over the logits.
    3. Compute CrossEntropyLoss for each window against `focused_target`.
    4. Take `min` (best matching window) or `mean` (if enforcing specific structure).
    5. Reference uses `min` over valid windows.

## 2. Gradient Normalization & Aggregation
**Problem**: Current impl accumulates raw gradients then averages. Reference normalizes *each* sample's gradient before accumulation.
**Solution**:
-   `compute_gradient` returns raw gradient (or normalized? Reference normalizes outside).
-   Update `GreaterOptimizer` loop:
    ```python
    grad = compute_gradient(...)
    grad = grad / grad.norm(dim=-1, keepdim=True)
    accumulated_grads += grad
    ```

## 3. Sequential Increasing Strategy
**Problem**: Fixed control length limits optimization space.
**Solution**:
-   **Config**: `start_len`, `end_len`, `patience`.
-   **Loop**:
    -   Start with `control_len = start_len`.
    -   Optimize until loss plateaus (using `patience` counter).
    -   If `control_len < end_len`:
        -   Extend control (append placeholder/best token).
        -   Continue optimization.

## 4. Caching
**Problem**: Expensive forward passes for Selection.
**Solution**:
-   `GreaterOptimizer` maintains `self.loss_cache = { token_ids_tuple: loss }`.
-   Query cache before `select_and_update` or inside it.

## 5. Non-ASCII Filtering
**Logic**:
-   Pre-compute mask of ASCII-only tokens (printable).
-   In `propose_candidates`, set logits of non-mask tokens to `-inf`.
