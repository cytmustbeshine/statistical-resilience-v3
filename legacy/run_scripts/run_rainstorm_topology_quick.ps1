$Python = "D:\soft\Python310\python.exe"
$CodeDir = "D:\TrafficGNN\dstsgcn_code"
$RoadLinks = "D:\TrafficGNN\data\road_links.csv"
$BaseOut = "D:\TrafficGNN\outputs\experiments_topology_quick"

Set-Location $CodeDir
$env:PYTHONUNBUFFERED = "1"

& $Python .\run_experiments.py `
  --suite rainstorm_quick `
  --max-nodes 41 `
  --epochs 20 `
  --batch-size 64 `
  --dynamic-top-k 3 `
  --quality-gate-bias -1.0 `
  --adj-source corr `
  --output-root "$BaseOut\corr"

& $Python .\run_experiments.py `
  --suite rainstorm_quick `
  --max-nodes 41 `
  --epochs 20 `
  --batch-size 64 `
  --dynamic-top-k 3 `
  --quality-gate-bias -1.0 `
  --adj-source road `
  --adj-path $RoadLinks `
  --road-knn 3 `
  --output-root "$BaseOut\road"

& $Python .\run_experiments.py `
  --suite rainstorm_quick `
  --max-nodes 41 `
  --epochs 20 `
  --batch-size 64 `
  --dynamic-top-k 3 `
  --quality-gate-bias -1.0 `
  --adj-source mix `
  --adj-path $RoadLinks `
  --road-knn 3 `
  --road-weight 0.5 `
  --output-root "$BaseOut\mix"
