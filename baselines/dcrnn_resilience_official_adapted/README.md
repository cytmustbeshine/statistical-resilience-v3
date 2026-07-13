# DCRNN-Resilience official-code-adapted baseline

Source paper/repository: `Charles117/resilience_shenzhen`.

This directory is a high-fidelity PyTorch adaptation of the public TensorFlow
DCRNN portion. It preserves dual random-walk supports, Chebyshev diffusion,
DCGRU equations, inverse-sigmoid curriculum learning, inverse-scaled masked
RMSE, gradient clipping, and the published optimizer/schedule defaults.

It is not described as a complete reproduction of the paper because the
public repository states that the Shenzhen traffic data, sensor locations,
and part of the resilience dynamic-capturing code are private or unpublished.

Protocols:

- `--protocol official`: chronological 70/10/20 split; use 24 input/output
  steps and published hyperparameters when the data are hourly.
- `--protocol fair`: use this project's common split/history/horizon so model
  comparisons share the same prediction task.

The original downloaded source is archived at:
`D:\TrafficGNN\papers\resilience_shenzhen_official`.