# L4 Two-Dimension Traffic Resilience Study Protocol

## Status
Preregistered before L4 event-result inspection. Neural-network architecture and training loss remain frozen.

## Research question
Can traffic resilience be represented as two separately interpretable processes instead of one universal scalar performance factor?

## Dimensions

### Demand/service-volume dimension
- Observed variable: flow conditional normal score.
- Direction is not interpreted as operating efficiency.
- Lower-tail deficit represents unusually low realized service volume or demand.
- A two-sided flow surprise is reported only as context, not as the primary resilience loss.

### Operating-efficiency dimension
- Speed is the mandatory anchor and higher speed means better efficiency.
- Occupancy, when available, is orientation-reversed before modeling.
- Flow is prohibited from entering the efficiency factor.
- With speed only, the efficiency state is the conditional speed normal score directly; no latent-factor identification is claimed.
- With speed and occupancy, a train-only missing-data one-factor model may combine them, with speed loading constrained positive by sign identification.

## Missing modalities
Missing values remain NaN and are never filled with zero. A missing occupancy value does not invalidate an observed speed efficiency state. Bridge has no efficiency definition and remains a flow-only service-volume proxy.

## Instantaneous deficits
- Demand lower-tail deficit: train-only conditional lower-tail surprisal of the demand score.
- Efficiency deficit: train-only conditional lower-tail surprisal of the efficiency score.
- Posterior uncertainty is reported separately and is not multiplied into either deficit.
- Each dimension uses its own train q90 high-state and q99 extreme-state threshold.

## Event states
At each time, system demand and efficiency deficits form four mutually exclusive states:
1. both high;
2. demand/service-volume high only;
3. efficiency high only;
4. neither high.

No weighted scalar combination is selected in L4.

## Event-level resilience
Peak, mean, cumulative deficit, high/extreme duration, degradation time, recovery time and recovery sensitivity are calculated separately for demand and efficiency. Recovery uses train q75 and 12 consecutive steps as the primary rule, with q50/q75/q90 and 6/12/24 sensitivity.

## Train-only rules
Profiles, factor parameters, orientations, thresholds, clipping and bootstrap settings use training data only. Event labels are used only for external validation and segmentation. Event windows and lag are not adjusted to improve L4.

## Acceptance criteria
L4 is accepted only as a two-dimensional descriptive definition if:
- Rainstorm efficiency deficit is higher during the event;
- at least two Typhoon segments have positive efficiency effects;
- Typhoon efficiency is not materially weaker than preregistered speed-only;
- PEMS04 and PEMS08 efficiency high-state rates do not clearly deteriorate relative to speed-only;
- speed/occupancy loadings and posterior states are stable when occupancy is used;
- demand and efficiency retain distinct event interpretations.

Acceptance does not authorize a neural-network target. A later study must separately decide whether a vector target, multi-task target, or event-process loss is appropriate.

## Rejection criteria
Reject or restrict L4 if efficiency reverses on Rainstorm/Typhoon, occupancy destabilizes PEMS, the dimensions cannot be interpreted separately, or a useful result requires test-set weighting or scalar recombination.