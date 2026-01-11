# Task List

## Traceability Improvements
- [ ] Add detailed logging to `greater_core.py` (Gradients, Token selections). <!-- id: core-logging -->
- [ ] Add logging to `greater.py` to trace coordinate update decisions. <!-- id: opt-logging -->

## Verification Run
- [ ] Setup `experiments/run_bbh_mini.py` for a 1-sample, 1-iteration "Deep Trace" run. <!-- id: mini-trace -->
- [ ] Run the mini-trace and capture full logs. <!-- id: execute-mini -->
- [ ] Analyze logs to ensure Gradients are propagating and selections are logical. <!-- id: analyze-reliability -->

## Scaling
- [ ] Prepare `experiments/run_bbh_full.py` for deployment to richer resources (8-16 samples, 20+ iterations). <!-- id: prep-full-run -->
