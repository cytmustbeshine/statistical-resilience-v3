param(
    [string]$DestinationRoot = "D:\TrafficGNN\backups\codex_chat"
)
$ErrorActionPreference = "Stop"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$destination = Join-Path $DestinationRoot $stamp
New-Item -ItemType Directory -Force -Path $destination | Out-Null
$codex = Join-Path $env:USERPROFILE ".codex"
Copy-Item (Join-Path $codex "sessions") (Join-Path $destination "sessions") -Recurse -Force
Copy-Item (Join-Path $codex "archived_sessions") (Join-Path $destination "archived_sessions") -Recurse -Force
foreach ($name in @("session_index.jsonl", "state_5.sqlite", "state_5.sqlite-wal", "state_5.sqlite-shm", ".codex-global-state.json")) {
    $source = Join-Path $codex $name
    if (Test-Path -LiteralPath $source) { Copy-Item -LiteralPath $source -Destination $destination -Force }
}
$thread = Get-ChildItem (Join-Path $codex "sessions") -Recurse -Filter "*019e69cf-8c18-7cd2-9a79-e58447ea29ba.jsonl" | Select-Object -First 1
$exporter = "D:\TrafficGNN\dstsgcn_code\scripts\export_codex_thread.py"
if ($thread -and (Test-Path -LiteralPath $exporter)) {
    & "D:\soft\Python310\python.exe" $exporter $thread.FullName (Join-Path $destination "traffic_thesis_conversation.md")
}
Write-Output "Codex backup completed: $destination"