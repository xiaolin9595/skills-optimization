# Design: Reliability Verification

## Enhanced Logging Trace
We will add `logger.debug` statements in `greater_core.py` to output:
- Number of candidates proposed.
- Min/Max gradient values.
- Selected candidate token IDs and their decoded strings.
- Loss values for all $\mu$ candidates in the selection stage.

## Focused Loss Diagnostics
One of the most complex parts is the "Unfolding" window in `compute_gradient`. We will add a verification step (or log) that confirms if the `focused_target` was actually found in the `loss_slice` tokens.

## Deployment Strategy
The scripts in `experiments/` are designed to be run from the root using `PYTHONPATH=src`.
To support "resource-rich servers", the configuration should be easily portable (already the case via `OptimizeConfig`).

## Process Handover
Since some processes seem to be "running" for hours, we will verify if they are actually active by checking if they update any log files. We will encourage explicit log file redirection (`> output.log 2>&1`).
