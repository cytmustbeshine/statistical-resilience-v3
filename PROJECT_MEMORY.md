# PROJECT MEMORY

## Research question
Unknown-disruption traffic resilience prediction under event-scarce training data. Estimate train-only conditional normal distributions, construct a unified latent traffic-performance index, and later study tail-risk-aware forecasting and uncertainty.

## Frozen architecture
DGCN-style dynamic graph + STSGCN-style synchronous convolution + existing quality fusion. Do not add graph gates, OOD scales, weather/event graphs, CVaR, or resilience heads until the statistical target is validated.

## Current statistical decision
Flow-only is valid for Bridge, both flow and speed are informative for Rainstorm, and speed is the primary signal for Typhoon. Flow-only and P3b softmax are rejected as universal labels. The next candidate is L3: train-only conditional-ECDF normal scores plus a missing-modality one-factor latent performance model.

## Authoritative outputs
- D:\TrafficGNN\outputs\statistical_resilience_profile_n41
- D:\TrafficGNN\outputs\flow_speed_resilience_analysis

## Important corrected facts
- PEMS test high-state rates: P3b PEMS04 6.50%, PEMS08 17.11%; speed-only PEMS04 8.77%, PEMS08 13.58%.
- Typhoon: 70 flow columns and 45 speed columns overall; only 16 of the first 41 flow nodes have matched speed.
- ECDF fallback must be audited by source level before L3. Current boolean fallback mainly indicates cell ECDF was unavailable; it does not prove parametric fallback.
- Bridge has one observed modality and cannot independently identify a full one-factor measurement model.

## Next step
L3-0 only: audit multilevel ECDF source, Typhoon speed anomalies, and provide an explicit conditional-CDF query API. Do not modify the neural network yet.
## Git workflow
1. Before editing code, inspect `git status` and preserve unrelated user changes.
2. Complete one coherent task, run the relevant compilation and tests, and inspect the diff before committing.
3. After verification, commit the task with a descriptive message and push it to the private `origin` repository.
4. Do not commit raw traffic data, model checkpoints, large generated arrays, experiment-output directories, credentials, tokens, or private keys.
5. Do not mark failing or unverified code as a stable version. Record any test limitation explicitly.
6. Never use destructive history operations or force-push unless the user explicitly requests them.
