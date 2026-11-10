param(
    [int]$RestartDelaySeconds = 30
)

$ErrorActionPreference = "Continue"

$mutexCreated = $false
$mutex = [System.Threading.Mutex]::new(
    $true,
    "TesseractFullValidation100M",
    [ref]$mutexCreated
)

if (-not $mutexCreated) {
    exit 0
}

$benchWindows = Split-Path -Parent $PSScriptRoot
$resultWindows = Join-Path $benchWindows "results\full_validation_100m"
$completeMarker = Join-Path $resultWindows "complete"
$watchdogLog = Join-Path $resultWindows "watchdog.log"
$benchWsl = "/mnt/c/Users/wjdrm/tesseract offloading/simulator/tesseract_msc_bench"

function Write-WatchdogLog {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss K"
    Add-Content -LiteralPath $watchdogLog -Value "$timestamp $Message"
}

try {
    New-Item -ItemType Directory -Path $resultWindows -Force | Out-Null
    Write-WatchdogLog "watchdog started"

    while (-not (Test-Path -LiteralPath $completeMarker)) {
        Write-WatchdogLog "starting or resuming full validation"

        & wsl.exe -d Ubuntu-24.04-D -- bash -lc (
            "cd '" + $benchWsl + "' && exec bash scripts/run_full_100m_validation.sh " +
            ">>results/full_validation_100m/driver.log 2>&1"
        )

        $validationExitCode = $LASTEXITCODE
        if (Test-Path -LiteralPath $completeMarker) {
            Write-WatchdogLog "validation complete; watchdog exiting"
            break
        }

        Write-WatchdogLog "validation exited with code $validationExitCode; restarting in $RestartDelaySeconds seconds"
        Start-Sleep -Seconds $RestartDelaySeconds
    }
}
catch {
    Write-WatchdogLog ("watchdog exception: " + $_.Exception.Message)
    throw
}
finally {
    if ($null -ne $mutex) {
        $mutex.ReleaseMutex()
        $mutex.Dispose()
    }
}
