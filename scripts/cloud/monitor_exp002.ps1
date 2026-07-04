# Local monitor for the two EXP-002 pods: first check after 5 min, then every
# 15 min for up to 7 hours. Emits PODWATCH:-prefixed lines on events (new
# tracebacks, phase transitions, completion, unreachable pods) for the agent's
# output watcher; plain status lines otherwise.
$pods = @(
    @{ Name = 'A'; Ip = '74.2.96.19';     Port = 11316 },
    @{ Name = 'B'; Ip = '157.157.221.29'; Port = 46156 }
)
$key = "$env:USERPROFILE\.ssh\runpod_aware"
$prevTb    = @{ A = -1;     B = -1 }
$prevPhase = @{ A = '';     B = '' }
$doneSeen  = @{ A = $false; B = $false }
$firstSeen = @{ A = $false; B = $false }
$deadline = (Get-Date).AddHours(7)
$interval = 300  # first check at +5 min, then 900s

while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds $interval
    $interval = 900
    foreach ($p in $pods) {
        if ($doneSeen[$p.Name]) { continue }
        $remote = "grep -cE 'Traceback' /workspace/exp002.log 2>/dev/null; grep -c 'run_id=' /workspace/exp002.log 2>/dev/null; grep -E 'PHASE|REALLY_ALL_DONE' /workspace/exp002.log 2>/dev/null | tail -n 1; tail -n 1 /workspace/exp002.log 2>/dev/null"
        $out = ssh -i $key -p $p.Port -o ConnectTimeout=15 -o BatchMode=yes "root@$($p.Ip)" $remote 2>$null
        $stamp = Get-Date -Format 'HH:mm'
        if (-not $out) {
            Write-Output "PODWATCH:ALERT pod $($p.Name) unreachable at $stamp - network blip or pod stopped (check RunPod console/API)"
            continue
        }
        $lines = @($out)
        $tb = 0; $runs = 0; $phase = ''; $last = ''
        if ($lines.Count -ge 1) { $tb    = [int]$lines[0] }
        if ($lines.Count -ge 2) { $runs  = [int]$lines[1] }
        if ($lines.Count -ge 3) { $phase = "$($lines[2])" }
        if ($lines.Count -ge 4) { $last  = "$($lines[3])" }

        if ($phase -match 'REALLY_ALL_DONE') {
            $doneSeen[$p.Name] = $true
            Write-Output "PODWATCH:DONE pod $($p.Name) complete at $stamp ($runs/9 runs logged)"
            continue
        }
        if (-not $firstSeen[$p.Name]) {
            $firstSeen[$p.Name] = $true
            Write-Output "PODWATCH:EVENT first check pod $($p.Name) at ${stamp}: tracebacks=$tb runs=$runs/9 phase='$phase' last='$last'"
        }
        elseif ($prevTb[$p.Name] -ge 0 -and $tb -gt $prevTb[$p.Name]) {
            Write-Output "PODWATCH:ALERT pod $($p.Name) NEW tracebacks ($($prevTb[$p.Name]) -> $tb) at ${stamp}: $last"
        }
        elseif ($phase -and $phase -ne $prevPhase[$p.Name]) {
            Write-Output "PODWATCH:EVENT pod $($p.Name) phase change at ${stamp}: $phase"
        }
        else {
            Write-Output "podcheck $($p.Name) ${stamp}: tb=$tb runs=$runs/9 last='$last'"
        }
        $prevTb[$p.Name] = $tb
        $prevPhase[$p.Name] = $phase
    }
    if ($doneSeen['A'] -and $doneSeen['B']) {
        Write-Output 'PODWATCH:DONE both pods complete - ready for pickup and adjudication'
        break
    }
}
Write-Output 'PODWATCH:EXIT monitor finished'
