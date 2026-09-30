param(
    [switch]$Cuda,
    [switch]$Test,
    [switch]$Asan,
    [switch]$AllowUnsupportedCudaCompiler,
    [ValidateSet('Debug', 'Release')][string]$Configuration = 'Release',
    [string]$BuildDirectory = '',
    [string]$VisualStudio = 'C:\Program Files\Microsoft Visual Studio\18\Professional',
    [string]$CudaArchitectures = '86'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (-not $BuildDirectory) { $BuildDirectory = if ($Cuda) { 'build-cuda' } elseif ($Asan) { 'build-asan' } else { 'build' } }
$buildPath = [System.IO.Path]::GetFullPath((Join-Path $root $BuildDirectory))
$vcvars = Join-Path $VisualStudio 'VC\Auxiliary\Build\vcvars64.bat'
$ninja = Join-Path $VisualStudio 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
if (-not (Test-Path -LiteralPath $vcvars)) { throw "MSVC environment script not found: $vcvars" }
if (-not (Test-Path -LiteralPath $ninja)) { throw "Ninja not found: $ninja" }
# Import the developer environment into this process; never print its contents.
$environmentLines = & $env:ComSpec /d /c "`"$vcvars`" >nul && set"
if ($LASTEXITCODE -ne 0) { throw 'MSVC environment initialization failed' }
foreach ($line in $environmentLines) {
    if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
}
$cudaFlag = if ($Cuda) { 'ON' } else { 'OFF' }
$asanFlag = if ($Asan) { 'ON' } else { 'OFF' }
$overrideFlag = if ($AllowUnsupportedCudaCompiler) { 'ON' } else { 'OFF' }
& cmake -S $root -B $buildPath -G Ninja "-DCMAKE_MAKE_PROGRAM=$ninja" '-DCMAKE_CXX_COMPILER=cl' "-DCMAKE_BUILD_TYPE=$Configuration" "-DKAN_ENABLE_CUDA=$cudaFlag" "-DKAN_ENABLE_ASAN=$asanFlag" "-DKAN_ALLOW_UNSUPPORTED_CUDA_COMPILER=$overrideFlag" "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures"
if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed' }
& cmake --build $buildPath --parallel
if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
if ($Test) {
    & ctest --test-dir $buildPath --output-on-failure
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
}
