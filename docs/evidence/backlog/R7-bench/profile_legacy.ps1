# R7: Nsight Systems CUDA API / kernel summaries of the legacy calls, base vs R7.
# Usage: profile_legacy.ps1 <legacy_base.exe> <legacy_r7.exe> <tmp-dir>
# Writes nsys-api.csv and nsys-kernels.csv next to this script.
param([string]$Base, [string]$R7, [string]$Tmp)
$ErrorActionPreference = 'Stop'
$nsys = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.5.2\target-windows-x64\nsys.exe'
$out = $PSScriptRoot
$api = @('build,case,api,calls,total_ms,median_us')
$kernels = @('build,case,kernel,time_pct,instances,median_us')
foreach ($build in @(@('base', $Base), @('r7', $R7))) {
    foreach ($case in 0, 1) {
        $name = "$($build[0])-case$case"
        $report = Join-Path $Tmp $name
        & $nsys profile --trace=cuda --force-overwrite=true -o $report $build[1] 1 $case | Out-Null
        & $nsys stats --report cuda_api_sum,cuda_gpu_kern_sum --format csv --output $report "$report.nsys-rep" | Out-Null
        foreach ($row in Import-Csv "${report}_cuda_api_sum.csv") {
            $api += "$($build[0]),$case,$($row.Name),$($row.'Num Calls'),$([double]$row.'Total Time (ns)'/1e6),$([double]$row.'Med (ns)'/1e3)"
        }
        foreach ($row in Import-Csv "${report}_cuda_gpu_kern_sum.csv") {
            $kernel = $row.Name.Substring(0, [Math]::Min(100, $row.Name.Length)).Replace('"', "'")
            $kernels += "$($build[0]),$case,`"$kernel`",$($row.'Time (%)'),$($row.Instances),$([double]$row.'Med (ns)'/1e3)"
        }
    }
}
$api | Out-File -Encoding ascii (Join-Path $out 'nsys-api.csv')
$kernels | Out-File -Encoding ascii (Join-Path $out 'nsys-kernels.csv')
