param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "../../..")).Path
$mscRoot = Join-Path $repositoryRoot "third_party/magic-state-cultivation"
$venvRoot = Join-Path $mscRoot ".venv"
$venvPython = Join-Path $venvRoot "Scripts/python.exe"

if (-not (Test-Path -LiteralPath (Join-Path $mscRoot "requirements.txt"))) {
    throw "Pinned MSC source is missing: $mscRoot"
}
if (-not (Test-Path -LiteralPath $venvPython)) {
    & $Python -m venv $venvRoot
}
& $venvPython -m pip install --requirement (Join-Path $mscRoot "requirements.txt")
Write-Host "MSC Python environment ready: $venvPython"
