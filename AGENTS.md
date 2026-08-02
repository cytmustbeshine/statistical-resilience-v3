# Project Instructions

## Research scope
- The two-dimensional L4 statistical definition is frozen: flow lower-tail demand/service-volume deficit and speed lower-tail efficiency deficit remain separate.
- Do not modify the model architecture, graph fusion, training loss, CVaR, uncertainty weighting, or conformal prediction until the corrected final forecasting protocol passes its stage gate.
- E-L4-2B-S0 completed the M0 DCRNN and M1 DSTSGCN traffic-only engineering smoke under the R3 dual-space contract. Full E-L4-2B-S remains blocked because final frozen-L4 M2 auxiliary supervision and M3 tail-risk runners are not implemented. Do not run formal seeds, bootstrap or final model selection; the next step is a separately audited M2/M3 smoke implementation without changing `model.py`, the frozen L4 definition or event windows.
- Read `PROJECT_MEMORY.md`, `EXPERIMENT_LOG.md`, and `DECISION_LOG.md` before substantial project work.

## Git workflow
- Inspect `git status` before editing and preserve unrelated user changes.
- After each coherent code task, run the relevant compilation and tests and inspect the diff.
- When checks pass, commit the verified task with a descriptive message and push it to the private `origin` repository.
- If checks fail, the diff contains unrelated changes, or credentials/data/checkpoints may be included, stop before commit or push and report the issue.
- Never commit raw traffic data, model checkpoints, large generated arrays, output directories, credentials, tokens, or private keys.
- Never force-push or use destructive history operations unless the user explicitly requests them.
