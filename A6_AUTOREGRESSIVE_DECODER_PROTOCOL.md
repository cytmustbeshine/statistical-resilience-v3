# A6 DSTSGCN Autoregressive Decoder Protocol

## Objective

Improve pure-model temporal forecasting without using public DCRNN predictions. Keep the existing Laplace dynamic graph learner, graph fusion and DSTSGCN blocks unchanged; replace only the direct multi-horizon projection at the model boundary with a node-shared GRU autoregressive decoder.

## Model

1. The existing DSTSGCN backbone produces the final node state.
2. A shared GRUCell decodes one step at a time for every node.
3. The last observed traffic value initializes the decoder input.
4. The decoder output becomes the next-step decoder input.
5. No DCRNN weights, predictions or hidden states are used.

## Inputs and Loss

Use traffic history only, train-only correlation adjacency, Huber traffic loss and validation physical MAE checkpoint selection. Event labels, weather/event adjacency, L4 auxiliary supervision, CVaR and test metrics are prohibited during A6-0 selection.

## A6-0 Gate

Run seed 42 on Bridge flow, Rainstorm flow/speed and Typhoon flow/speed. The decoder advances only if it improves validation MAE over direct-head M1 in at least four of five tasks, mean MAE ratio is at most 0.97 and no task regresses more than 3%.

A6-0 failure stops this decoder design before any test evaluation.