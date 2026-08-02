# Project Instructions

## Research scope
- The two-dimensional L4 statistical definition is frozen: flow lower-tail demand/service-volume deficit and speed lower-tail efficiency deficit remain separate.
- E-L4-2B formal training is complete for M0-M3, three seeds and 60 tasks. Do not rerun seeds, tune on test events, or call any candidate the final thesis model.
- The definitive fair L4 evaluation is frozen-profile postprocessing of physical flow/speed predictions for every model. M2/M3 auxiliary-head archives are diagnostics only and must never replace `g_flow(Q_hat)` or `g_speed(V_hat)` in final metrics.
- The final stage decision is negative/partial: accept limited evidence for M2 auxiliary supervision (9/15 continuous-L4 improvements), reject the M3 CVaR increment (7/15 q90-tail improvements), reject A3-L4 superiority over DCRNN (3/15 traffic-MAE improvements), and select no final model.
- Keep the frozen L4 definition, model architecture, event windows and test thresholds unchanged unless a new stage is explicitly preregistered.
- Read `PROJECT_MEMORY.md`, `EXPERIMENT_LOG.md`, and `DECISION_LOG.md` before substantial project work.

## Git workflow
- Inspect `git status` before editing and preserve unrelated user changes.
- After each coherent code task, run the relevant compilation and tests and inspect the diff.
- When checks pass, commit the verified task with a descriptive message and push it to the private `origin` repository.
- If checks fail, the diff contains unrelated changes, or credentials/data/checkpoints may be included, stop before commit or push and report the issue.
- Never commit raw traffic data, model checkpoints, large generated arrays, output directories, credentials, tokens, or private keys.
- Never force-push or use destructive history operations unless the user explicitly requests them.
