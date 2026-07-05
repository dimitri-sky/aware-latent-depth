# Watch pod C for the Stage-1/2 sentinel (STAGE12_FINAL_DONE), every 15 min up
# to 6 hours. Unreachable = pod auto-stopped after completion.
$key = "$env:USERPROFILE\.ssh\runpod_aware"
for ($i = 0; $i -lt 24; $i++) {
    Start-Sleep -Seconds 900
    $out = ssh -i $key -p 30844 -o ConnectTimeout=15 -o BatchMode=yes root@153.68.16.232 "test -f /workspace/STAGE12_FINAL_DONE && echo DONE; tail -n 1 /workspace/stage12.log" 2>$null
    if (-not $out) {
        Write-Output 'PODWATCH4:DONE pod C unreachable - likely auto-stopped after completion'
        break
    }
    $lines = @($out)
    if ($lines[0] -eq 'DONE') {
        Write-Output 'PODWATCH4:DONE Stage-1/2 chain complete'
        break
    }
    Write-Output "podCstage12 $(Get-Date -Format HH:mm): $($lines[-1])"
}
Write-Output 'PODWATCH4:EXIT watcher finished'
