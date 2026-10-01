param(
    [string]$Executable = 'build-m2-bench/m2_benchmark.exe',
    [string]$Output = 'docs/evidence/m2/final.csv',
    [ValidateSet('all','cpu','legacy','resident')][string]$Backend = 'all',
    [int]$Case = -1,
    [int]$Warmups = 2,
    [int]$Repeats = 7,
    [string]$BaselineExecutable = ''
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$executablePath = [IO.Path]::GetFullPath((Join-Path $root $Executable))
$outputPath = [IO.Path]::GetFullPath((Join-Path $root $Output))
if (-not (Test-Path -LiteralPath $executablePath)) { throw "Missing benchmark: $executablePath" }
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outputPath) | Out-Null
if ($BaselineExecutable) {
    $baselinePath = [IO.Path]::GetFullPath((Join-Path $root $BaselineExecutable))
    if (-not (Test-Path -LiteralPath $baselinePath)) { throw "Missing baseline: $baselinePath" }
    $balancedRows = @()
    $cases = if ($Case -eq -1) { @(0,1,3) } else { @($Case) }
    foreach ($caseIndex in $cases) {
        $runIndex = 0
        foreach ($variantName in @('preopt','trial','trial','preopt','preopt','trial','trial','preopt')) {
            $variantPath = if ($variantName -eq 'preopt') { $baselinePath } else { $executablePath }
            $csvLines = & $variantPath --backend resident --case $caseIndex --warmups $Warmups --repeats $Repeats
            if ($LASTEXITCODE -ne 0) { throw "Balanced benchmark failed: exit $LASTEXITCODE" }
            foreach ($row in ($csvLines | ConvertFrom-Csv)) {
                $row | Add-Member -NotePropertyName variant -NotePropertyValue $variantName
                $row | Add-Member -NotePropertyName run -NotePropertyValue $runIndex
                $balancedRows += $row
            }
            $runIndex++
        }
    }
    $balancedRows | Export-Csv -NoTypeInformation -Encoding utf8 -LiteralPath $outputPath
} else {
    # Capture stdout only, preserving a failing process's stderr and exit status.
    & $executablePath --backend $Backend --case $Case --warmups $Warmups --repeats $Repeats |
        Set-Content -LiteralPath $outputPath -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw "Benchmark failed: exit $LASTEXITCODE" }
}
$rows = Import-Csv -LiteralPath $outputPath
if (-not $rows) { throw 'Benchmark produced no measurements' }
Get-FileHash -Algorithm SHA256 -LiteralPath $outputPath | Select-Object Path,Hash
$rows | Select-Object case,family,topology,batch,backend,median_ms,iqr_ms,max_abs_error | Format-Table -AutoSize
