# DECISION LOG

## Rejected: universal flow deficit
Reason: Typhoon flow effect is negative and its bootstrap interval crosses zero.

## Rejected: P3b softmax as universal label
Reason: weaker than speed-only on Typhoon and raises non-event high-state on Rainstorm/PEMS08.

## Rejected: lag adjustment to rescue Typhoon flow
Reason: flow and joint lag distributions are unstable; speed response is synchronous at lag 0.

## Active candidate: L3 latent traffic performance
Use train-only conditional ECDF normal scores and a missing-modality one-factor model. Validate offline before modifying DGCN-STSGCN.