# PowerShell shim for Makefile targets on the local Windows box.
# Usage: .\make.ps1 <target>
param([Parameter(Mandatory = $true)][string]$Target)

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

switch ($Target) {
    "data" {
        & $py scripts/make_data.py --split train --per-family 4000
        & $py scripts/make_data.py --split eval --per-family 400
    }
    "test"       { & $py -m pytest tests/ -q }
    "audit"      { & $py -m sage.contamination.audit --train-dir data/sage/train --eval-dir data/sage/eval }
    "validity"   { & $py scripts/validity_gate.py }
    "train_tiny" { & $py scripts/train_tiny.py --config experiments/configs/exp001_loop_falsifier.yaml }
    "eval"       { & $py scripts/eval.py }
    "ablate"     { & $py scripts/ablate.py }
    "report"     { & $py scripts/report.py }
    "aa"         { & $py scripts/aa_test.py }
    default      { Write-Error "Unknown target: $Target" }
}
