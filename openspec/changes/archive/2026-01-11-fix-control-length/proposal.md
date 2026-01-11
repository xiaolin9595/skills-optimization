# Proposal: Fix Control Length Mismatch

## Problem
The current implementation of the GreaTer optimization loop assumes that `config.start_len` and `config.end_len` correspond to the *actual* token count of the control prompt. However, it initializes the optimization with the raw skill content (e.g., "Let's think step by step.", approx 8 tokens) which may significantly exceed `start_len` (e.g., 2). This mismatch causes:
1.  **Sequential Increasing Failure**: The loop iterates `length_iter` from 2 to 4, adding "!" each time, resulting in a control prompt of length 8 -> 9 -> 10 instead of the intended 2 -> 3 -> 4.
2.  **Patience Logic Confusion**: The optimization loop operates on the assumption of a shorter prompt, potentially leading to inefficient usage of patience budgets and misaligned expectations.
3.  **Inconsistency with GreaTer**: The original GreaTer paper initializes with a fixed number of placeholder tokens (e.g., 20 "!") to strictly control the optimization space.

## Solution
We will implement an explicit control initialization and resizing logic within `GreaterOptimizer` to force alignment with `config.start_len`.

1.  **Initialize with Placeholders**: If `skill.content` is provided but we are strictly following sequential increasing from `start_len`, we should *ignore* or *truncate* the initial content to match `start_len` tokens if they differ significantly, OR optimally, we should initialize with `! ! ...` (placeholders) of exact `start_len` if the goal is pure GreaTer optimization.
    - *Decision*: Since the user might provide a "Warm Start" prompt, we should RESPECT it if it's close, but if we are in "Sequential Mode" starting from a small N, we must enforce N.
    - *Strategy*:
        - Tokenize `skill.content`.
        - If `len(tokens) != config.start_len`:
            - If `len > start_len`: Truncate to `start_len`.
            - If `len < start_len`: Pad with `!` to `start_len`.
        - *Alternative*: If `skill.content` is meant to be a seed, maybe we should set `start_len` = `len(seed)`.
        - *Refined Strategy used in Proposal*: We will enforce `start_len` by resizing the `skill.content`. If the user wants to keep "Let's think step by step", they should set `start_len` to ~8. If they set `start_len=2`, they imply they want to compress or start fresh.
        - *Proposed Implementation*: Inside `optimize`, before the loop, tokenize `current_control_text`. Resize it to `config.start_len` using truncation or padding with `!`.

2.  **Strict Length Control in Loop**:
    - Inside the `length_iter` loop, ensure the control text *exactly matches* `length_iter` tokens.
    - When incrementing length (`length_iter > prev`), append exactly *one* `!` token.

## Risks
- **Loss of Semantic Content**: Truncating "Let's think step by step" to 2 tokens might leave "Let's think" or just "Let's", destroying the prompt's initial utility.
    - *Mitigation*: Log a warning if truncation is severe.
    - *Mitigation*: The GreaTer method inherently assumes we optimized *from scratch* (or from a random/set initialization) rather than refining a long semantic prompt. "Gradient based Prompt Optimization" typically starts with placeholders.
    - *User Education*: Docs should clarify that for refinement, `start_len` should match input length.

## Delimitations
- We focus on `GreaterOptimizer.optimize` logic.
- We do not change `run_gsm8k.py` user input, but we make the optimizer handle it gracefully (by enforcing the config).
