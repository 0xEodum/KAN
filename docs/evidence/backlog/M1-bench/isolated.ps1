# Isolated reruns of ABBA outlier cases, alternating builds (base, m1) for
# three rounds. Usage: isolated.ps1 <base-dir> <m1-dir> <output-csv>
param([string]$Base, [string]$M1, [string]$Output)
$ErrorActionPreference = 'Stop'
$cases = @(@('m2', 6, 'cpu'), @('m2', 21, 'cpu'), @('m2', 15, 'cpu'), @('m4', 8, 'cpu'), @('m4', 3, 'resident'), @('m4', 5, 'resident'))
$rows = @()
foreach ($round in 0..2) {
    foreach ($c in $cases) {
        foreach ($tag in 'base', 'm1') {
            $dir = if ($tag -eq 'base') { $Base } else { $M1 }
            $exe = Join-Path $dir "$($c[0])_benchmark.exe"
            $lines = & $exe --backend $c[2] --case $c[1] --warmups 2 --repeats 15
            if ($LASTEXITCODE -ne 0) { throw "failed $($c -join ' ') $tag" }
            foreach ($row in ($lines | ConvertFrom-Csv)) {
                $rows += [pscustomobject]@{ suite = $c[0]; case = $c[1]; backend = $row.backend; build = $tag; round = $round; median_ms = $row.median_ms }
            }
        }
    }
}
$rows | Export-Csv -NoTypeInformation -LiteralPath $Output
$rows | Group-Object suite, case, backend, build | ForEach-Object {
    $m = $_.Group | ForEach-Object { [double]$_.median_ms }
    '{0}: {1}' -f $_.Name, (($m | ForEach-Object { $_.ToString('0.000') }) -join ' ')
}
