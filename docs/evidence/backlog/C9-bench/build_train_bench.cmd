@rem Usage: build_train_bench.cmd <build-dir> <output-exe> [C9]
@rem Links train_bench.cpp against a Release CUDA build made by scripts\build.ps1 -Cuda.
@rem The third argument C9 enables the train_step mode (C9 builds only).
@setlocal
@set ROOT=%~dp0..\..\..\..
@set DEFS=
@if "%3"=="C9" set DEFS=/DKAN_C9_API
@call "C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul
@set LIBS="%ROOT%\%1\kan_cuda.lib" "%ROOT%\%1\kan.lib" "%CUDA_PATH%\lib\x64\cudart_static.lib" "%CUDA_PATH%\lib\x64\cublas.lib"
cl /nologo /std:c++20 /O2 /EHsc /MD %DEFS% /I "%ROOT%\include" "%~dp0train_bench.cpp" %LIBS% /Fo:"%TEMP%\\" /Fe:"%2"
