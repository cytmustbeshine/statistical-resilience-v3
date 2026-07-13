param(
    [string]$Python = "D:\soft\Python310\python.exe",
    [string]$CodeDir = "D:\TrafficGNN\dstsgcn_code",
    [int]$Epochs = 20,
    [int]$MaxNodes = 41,
    [string]$WeatherOutputRoot = "D:\TrafficGNN\outputs\thesis_main_weather_e20",
    [string]$NormalOutputRoot = "D:\TrafficGNN\outputs\thesis_main_normal_e20",
    [string]$ReportOutputDir = "D:\TrafficGNN\outputs\thesis_report_e20"
)

$ErrorActionPreference = "Stop"

Write-Host "[Thesis] Compile check"
& $Python -m py_compile `
    "$CodeDir\run_experiments.py" `
    "$CodeDir\analyze_weather_shift_slices.py" `
    "$CodeDir\analyze_thesis_results.py" `
    "$CodeDir\train.py" `
    "$CodeDir\model.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[Thesis] Run weather seeds"
& $Python "$CodeDir\run_experiments.py" `
    --suite thesis_weather_seeds `
    --models quality_v2,quality_stat_shift_auto_v2 `
    --epochs $Epochs `
    --max-nodes $MaxNodes `
    --output-root $WeatherOutputRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[Thesis] Run normal seeds"
& $Python "$CodeDir\run_experiments.py" `
    --suite thesis_normal_seeds `
    --models quality_v2,quality_stat_shift_auto_v2 `
    --epochs $Epochs `
    --max-nodes $MaxNodes `
    --output-root $NormalOutputRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$weatherSummary = Join-Path $WeatherOutputRoot "thesis_weather_seeds\experiment_summary.csv"
$normalSummary = Join-Path $NormalOutputRoot "thesis_normal_seeds\experiment_summary.csv"
$sliceDir = Join-Path $WeatherOutputRoot "slice_analysis"

Write-Host "[Thesis] Weather slice analysis"
& $Python "$CodeDir\analyze_weather_shift_slices.py" `
    --summary-csv $weatherSummary `
    --base-model quality_v2 `
    --shift-model quality_stat_shift_auto_v2 `
    --output-dir $sliceDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[Thesis] Thesis result analysis"
& $Python "$CodeDir\analyze_thesis_results.py" `
    --weather-summary $weatherSummary `
    --normal-summary $normalSummary `
    --slice-report-dir $sliceDir `
    --output-dir $ReportOutputDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[Thesis] Final report:"
Write-Host (Join-Path $ReportOutputDir "thesis_result_report.md")
