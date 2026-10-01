@rem Usage: build_resident_bench.cmd [build-dir]  (default build-m4-cuda, needs a Release CUDA build)
@setlocal
@set ROOT=%~dp0..\..\..
@set BUILD=%1
@if "%BUILD%"=="" set BUILD=build-m4-cuda
@call "C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /std:c++20 /O2 /EHsc /MD /I "%ROOT%\include" "%~dp0resident_bench.cpp" "%ROOT%\%BUILD%\kan_cuda.lib" "%ROOT%\%BUILD%\kan.lib" "%CUDA_PATH%\lib\x64\cudart_static.lib" /Fo:"%TEMP%\\" /Fe:"%TEMP%\resident_bench.exe"
