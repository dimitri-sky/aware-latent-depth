# Watch pod B for the EXP-003 sentinel (EXP003_FINAL_DONE), checking every
# 15 minutes for up to 8 hours. Unreachable = pod auto-stopped after completion.
$key = "$env:USERPROFILE\.ssh\runpod_aware"
for ($i = 0; $i -lt 32; $i++) {
    Start-Sleep -Seconds 900
    $out = ssh -i $key -p 46156 -o ConnectTimeout=15 -o BatchMode=yes root@157.157.221.29 "test -f /workspace/EXP003_FINAL_DONE && echo DONE; tail -n 1 /workspace/exp003.log" 2>$null
    if (-not $out) {
        Write-Output 'PODWATCH3:DONE pod B unreachable - likely auto-stopped after completion'
        break
    }
    $lines = @($out)
    if ($lines[0] -eq 'DONE') {
        Write-Output 'PODWATCH3:DONE EXP-003 chain complete'
        break
    }
    Write-Output "podBexp003 $(Get-Date -Format HH:mm): $($lines[-1])"
}
Write-Output 'PODWATCH3:EXIT watcher finished'
