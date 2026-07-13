# Project Instructions

## Research scope
- Treat DGCN-STSGCN as a frozen forecasting backbone until the statistical resilience target is validated.
- Do not modify the model architecture, graph fusion, training loss, CVaR, uncertainty weighting, or conformal prediction unless the user explicitly starts that phase.
- The current next step is L3-0: audit hierarchical ECDF source levels, Typhoon speed anomalies, and the conditional-CDF query API before implementing the latent factor model.
- Read `PROJECT_MEMORY.md`, `EXPERIMENT_LOG.md`, and `DECISION_LOG.md` before substantial project work.

## Git workflow
- Inspect `git status` before editing and preserve unrelated user changes.
- After each coherent code task, run the relevant compilation and tests and inspect the diff.
- When checks pass, commit the verified task with a descriptive message and push it to the private `origin` repository.
- If checks fail, the diff contains unrelated changes, or credentials/data/checkpoints may be included, stop before commit or push and report the issue.
- Never commit raw traffic data, model checkpoints, large generated arrays, output directories, credentials, tokens, or private keys.
- Never force-push or use destructive history operations unless the user explicitly requests them.