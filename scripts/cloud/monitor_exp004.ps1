# Watch an EXP-004/009 pod for the session FINAL_DONE marker, every 15 min, up to 9.5h.
# Usage: powershell -File monitor_exp004.ps1 -SshPort 12345 -SshHost 1.2.3.4 -Session A
param(
    [Parameter(Mandatory = $true)][int]$SshPort,
    [Parameter(Mandatory = $true)][string]$SshHost,
    [ValidateSet('A', 'B')][string]$Session = 'A'
)
$key = "$env:USERPROFILE\.ssh\runpod_aware"
$marker = "/workspace/EXP004${Session}_FINAL_DONE"
$log = "/workspace/exp004$($Session.ToLower()).log"
for ($i = 0; $i -lt 38; $i++) {
    Start-Sleep -Seconds 900
    $out = ssh -i $key -p $SshPort -o ConnectTimeout=15 -o BatchMode=yes "root@$SshHost" "test -f $marker && echo DONE; tail -n 1 $log" 2>$null
    if (-not $out) {
        Write-Output "PODWATCH-EXP004${Session}:DONE pod unreachable - likely auto-stopped after completion"
        break
    }
    $lines = @($out)
    if ($lines[0] -eq 'DONE') {
        Write-Output "PODWATCH-EXP004${Session}:DONE session chain complete"
        break
    }
    Write-Output "exp004$Session $(Get-Date -Format HH:mm): $($lines[-1])"
}
Write-Output "PODWATCH-EXP004${Session}:EXIT watcher finished"
