@rem Usage: build_large_step.cmd <build-dir> <output-exe>
@rem Links large_step.cpp against a Release CUDA build made by scripts\build.ps1 -Cuda.
@setlocal
@set ROOT=%~dp0..\..\..\..
@call "C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul
@set LIBS="%ROOT%\%1\kan_cuda.lib" "%ROOT%\%1\kan.lib" "%CUDA_PATH%\lib\x64\cudart_static.lib"
@if exist "%CUDA_PATH%\lib\x64\cublas.lib" set LIBS=%LIBS% "%CUDA_PATH%\lib\x64\cublas.lib" "%CUDA_PATH%\lib\x64\cublasLt.lib"
cl /nologo /std:c++20 /O2 /EHsc /MD /I "%ROOT%\include" "%~dp0large_step.cpp" %LIBS% /Fo:"%TEMP%\\" /Fe:"%2"
