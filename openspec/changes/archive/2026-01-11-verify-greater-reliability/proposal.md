# Proposal: Verify GreaTer Reliability (End-to-End)

## Problem
We have implemented the GreaTer optimization flow, including:
- `SkillPrompter` with reasoning and extraction support.
- `GreaterOptimizer` with sequential increasing and focused loss.
- `GreaterCore` with candidate proposal and gradient computation.
- Support for Llama-3 templates.
- Support for BBH and GSM8K data loading.

We need to verify that this entire flow is **reliable** and **functional** when running with a real model (Llama-3.2-3B). Reliability means:
1.  The model loads correctly and runs on the available hardware (GPU).
2.  The candidate proposal stage successfully generates tokens.
3.  The reasoning generation stage produces coherent text.
4.  The gradient calculation stage produces non-zero gradients.
5.  The selection stage successfully identifies candidates that reduce loss.
6.  The overall loss decreases over iterations (or at least the mechanism executes without error).

## Goals
1.  Establish a "Mini-Verify" baseline using 4-8 samples from BBH.
2.  Use the local Llama-3.2-3B model.
3.  Log detailed step-by-step progress (Gradients, Loss, Selected Tokens).
4.  Verify that the optimized prompt is different from the original and shows potential improvement.

## Solution
1.  **Refine Logging**: Ensure `GreaterOptimizer` and `GreaterCore` have sufficient `DEBUG`/`INFO` logs to trace the 4-stage process.
2.  **Mini-Run**: Execute `experiments/run_bbh.py` with minimal iterations to confirm end-to-end execution.
3.  **Correctness Check**: Validate that `focused_loss` is correctly identifying the target answer in the generated text.

## Risks
- **Hardware Constraints**: 3B model might still be slow or cause OOM if not handled carefully (though 3B should fit in ~8-12GB VRAM).
- **Stuck Processes**: Long run times in background processes suggest potential issues with multi-gpu or synchronization in the current environment's `GreaterOptimizer`.
