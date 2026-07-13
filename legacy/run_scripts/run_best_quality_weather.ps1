$Python = "D:\soft\Python310\python.exe"
$CodeDir = "D:\TrafficGNN\dstsgcn_code"
$RainstormCsv = "D:\TrafficGNN\data\rainstorm_traffic_state.csv"
$OutDir = "D:\TrafficGNN\outputs\rainstorm_static_prior_quality_weather_n30_e20_k3_b-1"
$WeatherCols = "altimeter,air_temp,relative_humidity,wind_speed,precip_accum_one_hour,visibility"

Set-Location $CodeDir
$env:PYTHONUNBUFFERED = "1"

& $Python train.py `
  --csv $RainstormCsv `
  --time-col "Time" `
  --value-suffix "_volume" `
  --extra-feature-cols $WeatherCols `
  --node-feature-suffixes "_speed" `
  --add-time-features `
  --history 12 `
  --horizon 12 `
  --hidden-dim 64 `
  --num-blocks 2 `
  --max-nodes 30 `
  --batch-size 64 `
  --epochs 20 `
  --fusion-mode fusion `
  --fusion-type quality `
  --dynamic-top-k 3 `
  --quality-gate-bias -1.0 `
  --seed 42 `
  --output-dir $OutDir

& $Python predict.py `
  --csv $RainstormCsv `
  --checkpoint "$OutDir\best_dstsgcn.pt" `
  --time-col "Time" `
  --value-suffix "_volume" `
  --extra-feature-cols $WeatherCols `
  --node-feature-suffixes "_speed" `
  --add-time-features `
  --history 12 `
  --horizon 12 `
  --hidden-dim 64 `
  --num-blocks 2 `
  --max-nodes 30 `
  --batch-size 64 `
  --fusion-mode fusion `
  --fusion-type quality `
  --dynamic-top-k 3 `
  --quality-gate-bias -1.0 `
  --output-dir "$OutDir\predictions" `
  --max-export-samples 200
