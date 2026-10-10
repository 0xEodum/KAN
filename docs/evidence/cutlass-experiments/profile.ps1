param([switch]$SystemsOnly,[switch]$ComputeOnly)
$ErrorActionPreference='Stop'
$root=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..'))
$exe=Join-Path $root 'build-cutlass-experiment\cutlass_experiment.exe'
if($env:KAN_EXPERIMENT_BINARY){$exe=$env:KAN_EXPERIMENT_BINARY}
$evidenceOut=$PSScriptRoot
if($env:KAN_EXPERIMENT_EVIDENCE){$evidenceOut=$env:KAN_EXPERIMENT_EVIDENCE}
$out=Join-Path $root 'build-cutlass-experiment\profiles'
New-Item -ItemType Directory -Force $out | Out-Null
$nsys='C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.5.2\target-windows-x64\nsys.exe'
$selection=Get-Content (Join-Path $evidenceOut 'selection.json') -Raw | ConvertFrom-Json
if(-not $ComputeOnly){foreach($caseName in @('wide','large')) {
    foreach($mode in @(0,2,3,4,5)) {
        $tile=if($mode -eq 0){0}else{$selection."$mode".tile}
        $stem="nsys-$caseName-m$mode"
        & $nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --show-output=true --force-overwrite=true -o (Join-Path $out $stem) $exe bench $mode $tile f32 resident $caseName 1 0 1 5 *> (Join-Path $evidenceOut "$stem.log")
        if($LASTEXITCODE -ne 0){throw "Nsight Systems failed: $stem"}
        & $nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv (Join-Path $out "$stem.nsys-rep") *> (Join-Path $evidenceOut "$stem-kernels.csv")
        if($LASTEXITCODE -ne 0){throw "Nsight Systems stats failed: $stem"}
    }
}}
if(-not $SystemsOnly){foreach($mode in @(2,4,5)) {
    $tile=$selection."$mode".tile
    $kernel=if($mode -eq 4){'regex:.*contract_kernel.*'}else{'regex:.*input_kernel.*'}
    $stem="ncu-wide-m$mode"
    & ncu --set full --kernel-name $kernel --launch-count 2 --force-overwrite --export (Join-Path $out $stem) $exe check $mode $tile f32 resident wide 1 0 *> (Join-Path $evidenceOut "$stem.log")
    if($LASTEXITCODE -ne 0){throw "Nsight Compute failed: $stem"}
    & ncu --import (Join-Path $out "$stem.ncu-rep") --csv --page details *> (Join-Path $evidenceOut "$stem-metrics.csv")
    if($LASTEXITCODE -ne 0){throw "Nsight Compute export failed: $stem"}
}
$stem='ncu-wide-virtual-dc'
& ncu --set full --kernel-name 'regex:.*contract_kernel.*' --launch-skip 3 --launch-count 1 --force-overwrite --export (Join-Path $out $stem) $exe check 4 $selection.'4'.tile f32 resident wide 1 0 *> (Join-Path $evidenceOut "$stem.log")
if($LASTEXITCODE -ne 0){throw 'Nsight Compute virtual dC failed'}
& ncu --import (Join-Path $out "$stem.ncu-rep") --csv --page details *> (Join-Path $evidenceOut "$stem-metrics.csv")
if($LASTEXITCODE -ne 0){throw 'Nsight Compute virtual dC export failed'}
foreach($tile in @(1,2)){
    $stem="ncu-deep-input-t$tile"
    & ncu --set full --kernel-name 'regex:.*input_kernel.*' --launch-count 2 --force-overwrite --export (Join-Path $out $stem) $exe check 2 $tile f32 resident deep 1 0 *> (Join-Path $evidenceOut "$stem.log")
    if($LASTEXITCODE -ne 0){throw "Nsight Compute adaptive tile failed: $stem"}
    & ncu --import (Join-Path $out "$stem.ncu-rep") --csv --page details *> (Join-Path $evidenceOut "$stem-metrics.csv")
    if($LASTEXITCODE -ne 0){throw "Nsight Compute adaptive export failed: $stem"}
}
}
