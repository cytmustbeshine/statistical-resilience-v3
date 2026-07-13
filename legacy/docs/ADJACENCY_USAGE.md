# Adjacency Usage

This project now supports three static adjacency sources:

```text
corr : Pearson correlation graph
road : road topology / geometry graph
mix  : road + correlation mixed graph
```

Prepared road topology file:

```text
D:\TrafficGNN\data\road_links.csv
```

The road file uses:

```text
link_id, from_node, to_node, geo_wkt
```

For selected traffic links, the code first tries link-to-link topology:

```text
upstream.to_node == downstream.from_node
```

If selected detector links are not directly adjacent, it falls back to a
geometry-based kNN road graph using `geo_wkt`.

## Quick topology comparison

Run this in VSCode PowerShell:

```powershell
cd D:\TrafficGNN\dstsgcn_code
powershell -ExecutionPolicy Bypass -File .\run_rainstorm_topology_quick.ps1
```

It runs:

```text
corr static / corr quality
road static / road quality
mix static  / mix quality
```

with:

```text
rainstorm dataset
41 nodes
seed=42
20 epochs
```

Summary files:

```text
D:\TrafficGNN\outputs\experiments_topology_quick\corr\rainstorm_quick\experiment_summary.csv
D:\TrafficGNN\outputs\experiments_topology_quick\road\rainstorm_quick\experiment_summary.csv
D:\TrafficGNN\outputs\experiments_topology_quick\mix\rainstorm_quick\experiment_summary.csv
```

## Manual examples

Correlation graph:

```powershell
D:\soft\Python310\python.exe .\run_experiments.py --suite rainstorm_quick --max-nodes 41 --adj-source corr
```

Road graph:

```powershell
D:\soft\Python310\python.exe .\run_experiments.py --suite rainstorm_quick --max-nodes 41 --adj-source road --adj-path D:\TrafficGNN\data\road_links.csv --road-knn 3
```

Mixed graph:

```powershell
D:\soft\Python310\python.exe .\run_experiments.py --suite rainstorm_quick --max-nodes 41 --adj-source mix --adj-path D:\TrafficGNN\data\road_links.csv --road-knn 3 --road-weight 0.5
```
