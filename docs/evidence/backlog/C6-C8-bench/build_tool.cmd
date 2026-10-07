@rem Usage: build_tool.cmd <absolute-build-dir> <source.cpp in this folder> <output-exe>
@rem Links a C6-C8 harness (rational_dump.cpp, rational_train_bench.cpp) against a
@rem Release CUDA build made by scripts\build.ps1 -Cuda (kan_cuda.lib, kan.lib).
@setlocal
@set ROOT=%~dp0..\..\..\..
@call "C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /std:c++20 /O2 /EHsc /MD /I "%ROOT%\include" "%~dp0%2" "%1\kan_cuda.lib" "%1\kan.lib" "%CUDA_PATH%\lib\x64\cudart_static.lib" "%CUDA_PATH%\lib\x64\cublas.lib" /Fo:"%TEMP%\\" /Fe:"%3"
