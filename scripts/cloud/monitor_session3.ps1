# Watch the session-3 pod for SESSION3_FINAL_DONE, every 15 min up to 9 hours.
$key = "$env:USERPROFILE\.ssh\runpod_aware"
for ($i = 0; $i -lt 36; $i++) {
    Start-Sleep -Seconds 900
    $out = ssh -i $key -p 44193 -o ConnectTimeout=15 -o BatchMode=yes root@87.197.100.115 "test -f /workspace/SESSION3_FINAL_DONE && echo DONE; tail -n 1 /workspace/session3.log" 2>$null
    if (-not $out) {
        Write-Output 'PODWATCH5:DONE session3 pod unreachable - likely auto-stopped after completion'
        break
    }
    $lines = @($out)
    if ($lines[0] -eq 'DONE') {
        Write-Output 'PODWATCH5:DONE session3 chain complete'
        break
    }
    Write-Output "session3 $(Get-Date -Format HH:mm): $($lines[-1])"
}
Write-Output 'PODWATCH5:EXIT watcher finished'
