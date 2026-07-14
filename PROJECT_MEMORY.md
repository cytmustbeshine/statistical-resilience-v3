# PROJECT MEMORY

## Research question
Unknown-disruption traffic resilience prediction under event-scarce training data. Estimate train-only conditional normal distributions, construct a unified latent traffic-performance index, and later study tail-risk-aware forecasting and uncertainty.

## Frozen architecture
DGCN-style dynamic graph + STSGCN-style synchronous convolution + existing quality fusion. Do not add graph gates, OOD scales, weather/event graphs, CVaR, or resilience heads until the statistical target is validated.

## Current statistical decision
Flow-only remains valid as a Bridge service proxy, both flow and speed are informative for Rainstorm, and speed-only remains the strongest Typhoon signal. Universal flow-only, P3b softmax, and the L3 one-factor latent performance definition are rejected as universal labels.

## Authoritative outputs
- D:\TrafficGNN\outputs\statistical_resilience_profile_n41
- D:\TrafficGNN\outputs\flow_speed_resilience_analysis
- D:\TrafficGNN\outputs\latent_traffic_performance_l3

## Important corrected facts
- PEMS test high-state rates: P3b PEMS04 6.50%, PEMS08 17.11%; speed-only PEMS04 8.77%, PEMS08 13.58%.
- Typhoon: 70 flow columns and 45 speed columns overall; only 16 of the first 41 flow nodes have matched speed.
- ECDF fallback must be audited by source level before L3. Current boolean fallback mainly indicates cell ECDF was unavailable; it does not prove parametric fallback.
- Bridge has one observed modality and cannot independently identify a full one-factor measurement model.

## Next step
Stop before neural-network integration. If a new statistical study is approved, preregister L4 with separate demand-level and operating-efficiency factors. Do not modify the neural network yet.
## Git workflow
1. Before editing code, inspect `git status` and preserve unrelated user changes.
2. Complete one coherent task, run the relevant compilation and tests, and inspect the diff before committing.
3. After verification, commit the task with a descriptive message and push it to the private `origin` repository.
4. Do not commit raw traffic data, model checkpoints, large generated arrays, experiment-output directories, credentials, tokens, or private keys.
5. Do not mark failing or unverified code as a stable version. Record any test limitation explicitly.
6. Never use destructive history operations or force-push unless the user explicitly requests them.
## L3 completed decision (2026-07-15)
Stage L3-0 and L3 are complete. The auditable CDF hierarchy is cell -> node-daytype -> node -> global -> robust parametric -> unavailable. In the N41 audit, Typhoon flow and speed use node-daytype empirical ECDFs rather than an untracked parametric fallback. The five negative Typhoon speed observations occur outside the 16 matched speed nodes in the first-41-flow analysis subnet.

The missing-data one-factor implementation, posterior variance, train-only latent profile, event process, initialization diagnostics, natural-day blocked stability, and 1,000-repetition moving-block bootstrap all passed engineering verification. However, L3 is rejected as a universal traffic-performance definition:
- Rainstorm L3 Cliff's delta is -0.8850 with a fully negative median-difference bootstrap interval; event high-state overlap is 0.
- Typhoon L3 remains positive but is materially weaker than speed-only: delta 0.2346 versus 0.5514, and overlap 9.56% versus 44.16%.
- PEMS04/PEMS08 L3 test high-state rates are 17.24% and 18.70%, so normal-state stability is not improved.
- Rainstorm initialization posterior agreement falls to about 0.3898, and the single factor learns stable negative flow loadings versus positive speed loadings.

Do not connect L3 to the neural network. Keep the architecture and training loss frozen. If research continues, preregister L4 as two separate factors (demand level and operating efficiency); do not implement L4 automatically.