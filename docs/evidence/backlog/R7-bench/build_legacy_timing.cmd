@rem Usage: build_legacy_timing.cmd <build-dir> <output-exe>
@rem Links legacy_timing.cpp against a Release CUDA build made by scripts\build.ps1 -Cuda.
@setlocal
@set ROOT=%~dp0..\..\..\..
@call "C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /std:c++20 /O2 /EHsc /MD /I "%ROOT%\include" "%~dp0legacy_timing.cpp" "%ROOT%\%1\kan_cuda.lib" "%ROOT%\%1\kan.lib" "%CUDA_PATH%\lib\x64\cudart_static.lib" "%CUDA_PATH%\lib\x64\cublas.lib" /Fo:"%TEMP%\\" /Fe:"%2"
