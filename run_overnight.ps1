# Overnight training runner.
#
# Trains the baseline and the hybrid model on the full dataset, logging
# everything to a timestamped file so the run can be inspected afterwards.
#
# Usage:
#   .\run_overnight.ps1
#
# The laptop must remain plugged in. Sleep and hibernate are already disabled
# on AC power, and the display switching off does not interrupt computation.

$ErrorActionPreference = "Continue"
$env:TF_CPP_MIN_LOG_LEVEL = "3"
$env:PYTHONUNBUFFERED = "1"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$logDir = Join-Path $projectRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$stamp   = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $logDir "training_$stamp.log"
$doneFile = Join-Path $logDir "COMPLETED_$stamp.txt"

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $Message
    Write-Output $line
    Add-Content -Path $logFile -Value $line
}

Write-Log "Overnight training run starting."
Write-Log "Project: $projectRoot"
Write-Log "Log file: $logFile"
Write-Log ""

# Keep the system awake for the duration of the run. This requests that
# Windows treat the session as busy even with no user input.
$signature = @'
[DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
try {
    $power = Add-Type -MemberDefinition $signature -Name "Power" -Namespace "Win32" -PassThru
    # ES_CONTINUOUS (0x80000000) | ES_SYSTEM_REQUIRED (0x00000001)
    [void]$power::SetThreadExecutionState(0x80000001)
    Write-Log "Sleep suppression requested."
} catch {
    Write-Log "Could not request sleep suppression. Confirm the laptop is plugged in."
}

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Log "ERROR: virtual environment interpreter not found at $python"
    exit 1
}

$startTime = Get-Date
Write-Log "Launching training. Expect six to nine hours."
Write-Log "Command: python -m src.train --experiments baseline hybrid --epochs 50"
Write-Log ""

# Progress bars are filtered so the log stays readable.
& $python -m src.train --experiments baseline hybrid --epochs 50 2>&1 |
    ForEach-Object {
        $line = $_.ToString()
        if ($line -notmatch '^\s*\d+/\d+\s' -and
            $line -notmatch 'it/s\]' -and
            $line -notmatch '^\s*$' -and
            $line -notmatch 'oneDNN|absl::|RemoteException') {
            $clean = $line -replace '\x1b\[[0-9;]*m', ''
            Add-Content -Path $logFile -Value $clean
            Write-Output $clean
        }
    }

$exitCode = $LASTEXITCODE
$duration = (Get-Date) - $startTime

Write-Log ""
Write-Log "Training process finished."
Write-Log ("Elapsed: {0:hh\:mm\:ss}" -f $duration)

# Record the outcome in a file that is easy to find in the morning.
$summary = @()
$summary += "Run started:  $startTime"
$summary += "Run finished: $(Get-Date)"
$summary += ("Elapsed:      {0:hh\:mm\:ss}" -f $duration)
$summary += ""

$comparison = Join-Path $projectRoot "results\metrics\experiment_comparison.csv"
$mcnemar    = Join-Path $projectRoot "results\metrics\mcnemar_test.json"
$bestModel  = Join-Path $projectRoot "models\best_model.keras"

if (Test-Path $comparison) {
    $summary += "RESULTS"
    $summary += (Get-Content $comparison -Raw)
    $summary += ""
    Write-Log "Comparison table written."
} else {
    $summary += "No comparison table was produced. Inspect the log for errors."
    Write-Log "WARNING: no comparison table found."
}

if (Test-Path $mcnemar) {
    $summary += "SIGNIFICANCE TEST"
    $summary += (Get-Content $mcnemar -Raw)
    $summary += ""
}

if (Test-Path $bestModel) {
    $sizeMb = [math]::Round((Get-Item $bestModel).Length / 1MB, 1)
    $summary += "Best model saved: models\best_model.keras ($sizeMb MB)"
    Write-Log "Best model saved."
} else {
    $summary += "No model artefact was produced."
}

$summary | Set-Content -Path $doneFile
Write-Log "Summary written to $doneFile"

# Release the sleep suppression request.
try { [void]$power::SetThreadExecutionState(0x80000000) } catch { }

Write-Log "Done."
