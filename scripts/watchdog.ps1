# scripts/watchdog.ps1
# Keeps the v4-hires training alive. If the trainer exits for any reason other
# than finishing (sleep, driver reset, crash), this resumes it from last.pt.
# It cannot survive a power loss -- it dies with the machine -- but it covers
# everything that leaves the OS running.
#
#   powershell -ExecutionPolicy Bypass -File scripts\watchdog.ps1
#
# Stop it with Ctrl+C. Stopping the watchdog does NOT stop training.

$py   = "C:/Users/Harley/anaconda3/envs/yolo-urinalysis/python.exe"
$data = "C:/Users/Harley/UroLens/UroLens-5-hires/data_oversampled.yaml"
$run  = "runs/detect/urinalysis/v4-hires"
$log  = "$run/watchdog.log"

function Note($m) {
  $line = "[{0}] {1}" -f (Get-Date -Format "MM-dd HH:mm:ss"), $m
  Write-Host $line
  Add-Content -Path $log -Value $line -Encoding utf8
}

Note "watchdog started"

while ($true) {
  $alive = Get-Process python -ErrorAction SilentlyContinue |
           Where-Object { $_.WorkingSet64 -gt 800MB }

  if (-not $alive) {
    # finished? results.csv reaching the epoch target means leave it alone.
    $csv = "$run/results.csv"
    $done = 0
    if (Test-Path $csv) { $done = (Import-Csv $csv).Count }
    if ($done -ge 150) { Note "training complete at epoch $done -- watchdog exiting"; break }

    Note "trainer not running (last epoch $done) -- resuming"
    Start-Process -FilePath $py `
      -ArgumentList @("scripts/train.py","--data",$data,"--name","v4-hires","--device","0","--resume") `
      -WorkingDirectory (Get-Location).Path
    Start-Sleep -Seconds 180   # give it time to spin up before checking again
  }

  Start-Sleep -Seconds 60
}
