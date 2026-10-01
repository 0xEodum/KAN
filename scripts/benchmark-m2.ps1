param(
    [string]$Executable = 'build-m2-bench/m2_benchmark.exe',
    [string]$Output = 'docs/evidence/m2/final.csv',
    [ValidateSet('all','cpu','legacy','resident')][string]$Backend = 'all',
    [int]$Case = -1,
    [int]$Warmups = 2,
    [int]$Repeats = 7
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$executablePath = [IO.Path]::GetFullPath((Join-Path $root $Executable))
$outputPath = [IO.Path]::GetFullPath((Join-Path $root $Output))
if (-not (Test-Path -LiteralPath $executablePath)) { throw "Missing benchmark: $executablePath" }
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outputPath) | Out-Null
# Capture stdout only, preserving a failing process's stderr and exit status.
& $executablePath --backend $Backend --case $Case --warmups $Warmups --repeats $Repeats |
    Set-Content -LiteralPath $outputPath -Encoding utf8
if ($LASTEXITCODE -ne 0) { throw "Benchmark failed: exit $LASTEXITCODE" }
$rows = Import-Csv -LiteralPath $outputPath
if (-not $rows) { throw 'Benchmark produced no measurements' }
Get-FileHash -Algorithm SHA256 -LiteralPath $outputPath | Select-Object Path,Hash
$rows | Select-Object case,family,topology,batch,backend,median_ms,iqr_ms,max_abs_error | Format-Table -AutoSize
