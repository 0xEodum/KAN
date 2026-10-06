@rem Usage (from this directory): build.cmd <source.cu> <output-exe> [extra nvcc flags]
@call "C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
nvcc -allow-unsupported-compiler -O3 -std=c++20 -arch=sm_86 -I ..\..\..\..\..\src -I ..\..\..\..\..\include %3 %1 -o %2 -lcublas
