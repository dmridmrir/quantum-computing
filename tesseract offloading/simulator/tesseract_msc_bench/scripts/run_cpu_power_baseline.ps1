[CmdletBinding()]
param(
    [string]$Distro = 'Ubuntu-24.04',
    [int]$NumShots = 100000,
    [int]$CpuThreads = 16,
    [int]$IdleSamples = 5,
    [int]$SampleIntervalSeconds = 1,
    [string]$RunName = ('cpu_rapl_{0}' -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ($NumShots -le 0 -or $CpuThreads -le 0 -or $IdleSamples -lt 2) {
    throw 'NumShots/CpuThreads must be positive and IdleSamples must be at least 2.'
}

$benchRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$resultRoot = Join-Path $benchRoot 'results\power'
$resultDir = Join-Path $resultRoot $RunName
New-Item -ItemType Directory -Force -Path $resultDir | Out-Null

function Convert-ToWslPath([string]$Path) {
    $converted = & wsl.exe -d $Distro -- wslpath -a $Path
    if ($LASTEXITCODE -ne 0 -or -not $converted) {
        throw "Failed to convert path for WSL: $Path"
    }
    return ($converted | Select-Object -Last 1).Trim()
}

$linuxBenchRoot = Convert-ToWslPath $benchRoot
$linuxResultDir = Convert-ToWslPath $resultDir
$model = "$linuxBenchRoot/workloads/msc_grafted_d3_d7_100m/model.dem"
$syndromes = "$linuxBenchRoot/workloads/msc_grafted_d3_d7_100m/syndromes.01"
$observables = "$linuxBenchRoot/workloads/msc_grafted_d3_d7_100m/observables.01"

# Prefer the compact b8 workload when it is available; both inputs describe the
# same pinned accepted-shot population.
$b8Syndromes = Join-Path $benchRoot 'workloads\msc_grafted_d3_d7_100m\syndromes.b8'
$b8Observables = Join-Path $benchRoot 'workloads\msc_grafted_d3_d7_100m\observables.b8'
if ((Test-Path -LiteralPath $b8Syndromes) -and (Test-Path -LiteralPath $b8Observables)) {
    $syndromes = Convert-ToWslPath $b8Syndromes
    $observables = Convert-ToWslPath $b8Observables
}

$counterPaths = @(
    '\Energy Meter(RAPL_Package0_PKG)\Power',
    '\Energy Meter(RAPL_Package0_DRAM)\Power',
    '\Processor Information(_Total)\% Processor Utility'
)

function Get-PowerSample([double]$ElapsedSeconds, [string]$Phase) {
    $sample = Get-Counter -Counter $counterPaths -MaxSamples 1
    $values = @{}
    foreach ($entry in $sample.CounterSamples) {
        if ($entry.Path -match 'RAPL_Package0_PKG') {
            $values.pkg_w = $entry.CookedValue / 1000.0
        } elseif ($entry.Path -match 'RAPL_Package0_DRAM') {
            $values.dram_w = $entry.CookedValue / 1000.0
        } elseif ($entry.Path -match '% Processor Utility') {
            $values.processor_utility_percent = $entry.CookedValue
        }
    }
    if (-not $values.ContainsKey('pkg_w')) {
        throw 'RAPL package power counter returned no sample.'
    }
    [pscustomobject]@{
        timestamp_utc = [DateTime]::UtcNow.ToString('o')
        elapsed_seconds = $ElapsedSeconds
        phase = $Phase
        package_w = [double]$values.pkg_w
        dram_w = if ($values.ContainsKey('dram_w')) { [double]$values.dram_w } else { $null }
        processor_utility_percent = if ($values.ContainsKey('processor_utility_percent')) {
            [double]$values.processor_utility_percent
        } else { $null }
    }
}

$samples = [System.Collections.Generic.List[object]]::new()
$idleWatch = [System.Diagnostics.Stopwatch]::StartNew()
for ($i = 0; $i -lt $IdleSamples; $i++) {
    $samples.Add((Get-PowerSample $idleWatch.Elapsed.TotalSeconds 'idle'))
    if ($i + 1 -lt $IdleSamples) {
        Start-Sleep -Seconds $SampleIntervalSeconds
    }
}
$idleWatch.Stop()

$stdoutPath = Join-Path $resultDir 'benchmark.stdout.log'
$stderrPath = Join-Path $resultDir 'benchmark.stderr.log'
$processInfo = [System.Diagnostics.ProcessStartInfo]::new()
$processInfo.FileName = 'wsl.exe'
$processInfo.UseShellExecute = $false
$processInfo.RedirectStandardOutput = $true
$processInfo.RedirectStandardError = $true
foreach ($arg in @(
    '-d', $Distro, '--', 'bash',
    "$linuxBenchRoot/scripts/run_cpu_dem_baseline.sh",
    $model, $syndromes, $observables, $linuxResultDir,
    $NumShots.ToString(), $CpuThreads.ToString()
)) {
    [void]$processInfo.ArgumentList.Add($arg)
}

$process = [System.Diagnostics.Process]::new()
$process.StartInfo = $processInfo
$runWatch = [System.Diagnostics.Stopwatch]::StartNew()
if (-not $process.Start()) {
    throw 'Failed to start WSL CPU benchmark.'
}
$stdoutTask = $process.StandardOutput.ReadToEndAsync()
$stderrTask = $process.StandardError.ReadToEndAsync()

while (-not $process.HasExited) {
    $samples.Add((Get-PowerSample $runWatch.Elapsed.TotalSeconds 'benchmark'))
    if (-not $process.WaitForExit($SampleIntervalSeconds * 1000)) {
        continue
    }
}
$process.WaitForExit()
$runWatch.Stop()
$stdoutTask.GetAwaiter().GetResult() | Set-Content -LiteralPath $stdoutPath -Encoding utf8
$stderrTask.GetAwaiter().GetResult() | Set-Content -LiteralPath $stderrPath -Encoding utf8
if ($process.ExitCode -ne 0) {
    throw "CPU benchmark failed with exit code $($process.ExitCode). See $stderrPath"
}

$samplePath = Join-Path $resultDir 'power_samples.csv'
$samples | Export-Csv -LiteralPath $samplePath -NoTypeInformation -Encoding utf8

$idle = @($samples | Where-Object phase -eq 'idle')
$active = @($samples | Where-Object phase -eq 'benchmark')
if ($active.Count -lt 2) {
    throw 'Benchmark was too short to obtain at least two active power samples.'
}

$benchmarkSummaryPath = Join-Path $resultDir 'summary.csv'
$benchmarkSummary = @{}
Import-Csv -LiteralPath $benchmarkSummaryPath | ForEach-Object {
    $benchmarkSummary[$_.metric] = $_.value
}
$completedShots = [double]$benchmarkSummary.completed_shots
$wallSeconds = [double]$benchmarkSummary.process_wall_seconds
$averagePackageW = ($active | Measure-Object package_w -Average).Average
$averageDramW = ($active | Where-Object dram_w -ne $null | Measure-Object dram_w -Average).Average
$idlePackageW = ($idle | Measure-Object package_w -Average).Average
$idleDramW = ($idle | Where-Object dram_w -ne $null | Measure-Object dram_w -Average).Average
$averagePackageAndDramW = $averagePackageW + $averageDramW
$idlePackageAndDramW = $idlePackageW + $idleDramW
$grossEnergyJ = $averagePackageW * $wallSeconds
$incrementalPowerW = [Math]::Max(0.0, $averagePackageW - $idlePackageW)
$incrementalEnergyJ = $incrementalPowerW * $wallSeconds
$grossPackageAndDramEnergyJ = $averagePackageAndDramW * $wallSeconds
$incrementalPackageAndDramPowerW = [Math]::Max(
    0.0, $averagePackageAndDramW - $idlePackageAndDramW
)
$incrementalPackageAndDramEnergyJ = $incrementalPackageAndDramPowerW * $wallSeconds
$processorName = Get-ItemPropertyValue `
    -LiteralPath 'HKLM:\HARDWARE\DESCRIPTION\System\CentralProcessor\0' `
    -Name ProcessorNameString

$summary = [ordered]@{
    result_kind = 'measured_windows_rapl_during_wsl_cpu_benchmark'
    run_name = $RunName
    processor = $processorName
    cpu_threads = $CpuThreads
    completed_shots = $completedShots
    process_wall_seconds = $wallSeconds
    process_wall_throughput_shots_per_s = $completedShots / $wallSeconds
    sample_interval_seconds = $SampleIntervalSeconds
    idle_sample_count = $idle.Count
    active_sample_count = $active.Count
    average_package_power_w = $averagePackageW
    average_dram_power_w = $averageDramW
    average_package_and_dram_power_w = $averagePackageAndDramW
    idle_average_package_power_w = $idlePackageW
    idle_average_dram_power_w = $idleDramW
    idle_average_package_and_dram_power_w = $idlePackageAndDramW
    incremental_package_power_w = $incrementalPowerW
    incremental_package_and_dram_power_w = $incrementalPackageAndDramPowerW
    gross_package_energy_j = $grossEnergyJ
    gross_package_and_dram_energy_j = $grossPackageAndDramEnergyJ
    incremental_package_energy_j = $incrementalEnergyJ
    incremental_package_and_dram_energy_j = $incrementalPackageAndDramEnergyJ
    gross_package_energy_j_per_shot = $grossEnergyJ / $completedShots
    gross_package_and_dram_energy_j_per_shot = $grossPackageAndDramEnergyJ / $completedShots
    incremental_package_energy_j_per_shot = $incrementalEnergyJ / $completedShots
    incremental_package_and_dram_energy_j_per_shot = $incrementalPackageAndDramEnergyJ / $completedShots
    counter_units = 'Get-Counter Energy Meter cooked values are milliwatts; divided by 1000'
    boundary = 'RAPL CPU package and DRAM domains; excludes PSU, motherboard, storage, and cooling'
}
$powerSummaryPath = Join-Path $resultDir 'power_summary.json'
$summary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $powerSummaryPath -Encoding utf8

Write-Output "CPU power baseline complete: $powerSummaryPath"
Write-Output ("Average package+DRAM power: {0:N2} W; gross energy: {1:N3} mJ/shot" -f `
    $averagePackageAndDramW, (($grossPackageAndDramEnergyJ / $completedShots) * 1000.0))
