param([string]$BuildDirectory='build-cutlass-experiment',[switch]$NoFma)
$ErrorActionPreference='Stop'
$root=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..'))
$buildPath=Join-Path $root $BuildDirectory
$vcvars='C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat'
$ninja='C:\Program Files\Microsoft Visual Studio\18\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
$environmentLines=& $env:ComSpec /d /c "`"$vcvars`" >nul && set"
if($LASTEXITCODE -ne 0){throw 'MSVC initialization failed'}
foreach($line in $environmentLines){if($line -match '^([^=]+)=(.*)$'){[Environment]::SetEnvironmentVariable($matches[1],$matches[2],'Process')}}
$fma=if($NoFma){'OFF'}else{'ON'}
& cmake -S $root -B $buildPath -G Ninja "-DCMAKE_MAKE_PROGRAM=$ninja" '-DCMAKE_CXX_COMPILER=cl' '-DCMAKE_BUILD_TYPE=Release' '-DKAN_ENABLE_CUDA=ON' '-DKAN_ALLOW_UNSUPPORTED_CUDA_COMPILER=ON' '-DCMAKE_CUDA_ARCHITECTURES=86' "-DKAN_CUDA_FMA=$fma" '-DKAN_CUTLASS_EXPERIMENT=ON' "-DKAN_CUTLASS_ROOT=$root/build-cutlass-deps/cutlass"
if($LASTEXITCODE -ne 0){throw 'configuration failed'}
& cmake --build $buildPath --parallel 4
if($LASTEXITCODE -ne 0){throw 'build failed'}
