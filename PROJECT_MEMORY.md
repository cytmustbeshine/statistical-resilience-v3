# PROJECT MEMORY

## Research question
Unknown-disruption traffic resilience prediction under event-scarce training data. Estimate train-only conditional normal distributions, construct a unified latent traffic-performance index, and later study tail-risk-aware forecasting and uncertainty.

## Frozen architecture
DGCN-style dynamic graph + STSGCN-style synchronous convolution + existing quality fusion. Do not add graph gates, OOD scales, weather/event graphs, CVaR, or resilience heads until the statistical target is validated.

## Current statistical decision
The supported statistical definition is two-dimensional: demand/service-volume deficit from flow and operating-efficiency deficit from speed, reported separately with event quadrants and separate recovery processes. Universal flow-only, P3b, L3 one-factor, scalar L4 recombination, and automatic occupancy inclusion are rejected or restricted.

## Authoritative outputs
- D:\TrafficGNN\outputs\statistical_resilience_profile_n41
- D:\TrafficGNN\outputs\flow_speed_resilience_analysis
- D:\TrafficGNN\outputs\latent_traffic_performance_l3
- D:\TrafficGNN\outputs\two_factor_traffic_resilience_l4

## Important corrected facts
- PEMS test high-state rates: P3b PEMS04 6.50%, PEMS08 17.11%; speed-only PEMS04 8.77%, PEMS08 13.58%.
- Typhoon: 70 flow columns and 45 speed columns overall; only 16 of the first 41 flow nodes have matched speed.
- ECDF fallback must be audited by source level before L3. Current boolean fallback mainly indicates cell ECDF was unavailable; it does not prove parametric fallback.
- Bridge has one observed modality and cannot independently identify a full one-factor measurement model.

## Next step
The offline descriptive definition is complete. Stop before neural-network integration. A new preregistered study must choose whether the network should predict a vector state, use multi-task supervision, or optimize event-process objectives.
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
## L4 two-dimension completed decision (2026-07-15)
The preregistered L4 study is complete. Traffic resilience is supported as a two-dimensional statistical description rather than one universal scalar:
- demand/service-volume state: flow conditional lower-tail deficit, interpreted as unusually low realized service volume or demand rather than operating efficiency;
- operating-efficiency state: speed conditional lower-tail deficit, with flow prohibited from entering this dimension.

Event quadrants resolve the previous contradiction. Rainstorm is predominantly both demand/service and efficiency loss (92.83% both-high during the main segment). Typhoon is predominantly efficiency-only loss: 40.08%, 40.51%, and 34.60% across its three segments, while demand-only/both-high rates are much smaller. Rainstorm efficiency delta is 0.9463. Typhoon efficiency deltas are positive in all three segments (0.5274, 0.5296, 0.4051).

The PEMS speed+reversed-occupancy efficiency extension is restricted/rejected for the core definition. Its bootstrap loadings are stable and positive, but test high-state rates deteriorate to 15.77% on PEMS04 and 17.13% on PEMS08, compared with speed-only 7.78% and 12.74%. Stable loadings do not override normal-state calibration failure.

Current recommended statistical definition:
1. report demand/service-volume deficit and operating-efficiency deficit separately;
2. report both-high, demand-only, efficiency-only, and neither event states;
3. compute peak, cumulative loss, duration, and recovery separately by dimension;
4. Bridge remains demand/service-volume only;
5. occupancy remains an auxiliary diagnostic, not a universal core efficiency input;
6. do not create a weighted scalar L4 score.

This completes the offline definition study at the descriptive statistical level, but does not authorize neural-network integration. A separate preregistered study is required to choose vector supervision, multi-task learning, or event-process objectives.