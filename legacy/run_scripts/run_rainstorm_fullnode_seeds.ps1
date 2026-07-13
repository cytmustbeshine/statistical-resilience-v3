$Python = "D:\soft\Python310\python.exe"
$CodeDir = "D:\TrafficGNN\dstsgcn_code"

Set-Location $CodeDir
$env:PYTHONUNBUFFERED = "1"

& $Python .\run_experiments.py `
  --suite rainstorm_seeds `
  --max-nodes 41 `
  --epochs 20 `
  --batch-size 64 `
  --dynamic-top-k 3 `
  --quality-gate-bias -1.0
