# Watch the EXP-008 pod for EXP008_FINAL_DONE, every 20 min up to 20 hours.
$key = "$env:USERPROFILE\.ssh\runpod_aware"
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 1200
    $out = ssh -i $key -p 30345 -o ConnectTimeout=15 -o BatchMode=yes root@153.68.16.232 "test -f /workspace/EXP008_FINAL_DONE && echo DONE; tail -n 1 /workspace/exp008.log" 2>$null
    if (-not $out) {
        Write-Output 'PODWATCH6:DONE exp008 pod unreachable - likely auto-stopped after completion'
        break
    }
    $lines = @($out)
    if ($lines[0] -eq 'DONE') {
        Write-Output 'PODWATCH6:DONE EXP-008 chain complete'
        break
    }
    Write-Output "exp008 $(Get-Date -Format HH:mm): $($lines[-1])"
}
Write-Output 'PODWATCH6:EXIT watcher finished'
