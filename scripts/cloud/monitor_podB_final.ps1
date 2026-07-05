# Watch pod B for the true final sentinel (EXP002_FINAL_DONE), checking every
# 10 minutes for up to 5 hours. Unreachable = pod auto-stopped after completion.
$key = "$env:USERPROFILE\.ssh\runpod_aware"
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 600
    $out = ssh -i $key -p 46156 -o ConnectTimeout=15 -o BatchMode=yes root@157.157.221.29 "grep -c EXP002_FINAL_DONE /workspace/exp002.log 2>/dev/null; tail -n 1 /workspace/exp002.log" 2>$null
    if (-not $out) {
        Write-Output 'PODWATCH2:DONE pod B unreachable - likely auto-stopped after completion'
        break
    }
    $lines = @($out)
    $count = 0
    if ($lines.Count -ge 1) { $count = [int]$lines[0] }
    if ($count -ge 1) {
        Write-Output 'PODWATCH2:DONE pod B truly complete'
        break
    }
    $last = ''
    if ($lines.Count -ge 2) { $last = $lines[1] }
    Write-Output "podBcheck $(Get-Date -Format HH:mm): $last"
}
Write-Output 'PODWATCH2:EXIT watcher finished'
