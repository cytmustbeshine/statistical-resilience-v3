# DECISION LOG

## Rejected: universal flow deficit
Reason: Typhoon flow effect is negative and its bootstrap interval crosses zero.

## Rejected: P3b softmax as universal label
Reason: weaker than speed-only on Typhoon and raises non-event high-state on Rainstorm/PEMS08.

## Rejected: lag adjustment to rescue Typhoon flow
Reason: flow and joint lag distributions are unstable; speed response is synchronous at lag 0.

## Active candidate: L3 latent traffic performance
Use train-only conditional ECDF normal scores and a missing-modality one-factor model. Validate offline before modifying DGCN-STSGCN.
## Rejected: L3 one-factor latent traffic performance (2026-07-15)
Engineering correctness was accepted, but the statistical definition was rejected.
Reasons:
1. Rainstorm event semantics reverse: L3 Cliff's delta -0.8850, bootstrap interval fully below zero, and event high-state overlap 0.
2. Typhoon L3 is clearly weaker than speed-only despite remaining positive.
3. PEMS04 and PEMS08 normal-state high-state rates are not improved and PEMS04 deteriorates sharply.
4. The one-factor model consistently learns negative flow loadings and positive speed/occupancy loadings, showing that demand level and operating efficiency are not one common performance axis.
5. Rainstorm has poor multi-initialization latent-state agreement, even though blocked-fold loading signs are stable.
6. Occupancy communalities near 1 on PEMS indicate that the factor is dominated by occupancy rather than a balanced universal performance construct.
Consequence: do not add L3 labels, auxiliary heads, CVaR, uncertainty weighting, conformal prediction, graph gates, or other network complexity. Keep Bridge as a flow-only service proxy and Typhoon speed-only as the strongest current event-performance signal. A future L4 study may separate demand level from operating efficiency, but L4 was not implemented in this stage.