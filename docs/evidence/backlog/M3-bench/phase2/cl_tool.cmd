@rem Usage: cl_tool.cmd <abs-build-dir> <abs-source.cpp> <output-exe> [cl flags]: links a harness against a Release CUDA build (headers from <build>..include).
@setlocal
@call "C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /std:c++20 /O2 /EHsc /MD %4 %5 /I "%1\..\include" "%2" "%1\kan_cuda.lib" "%1\kan.lib" "%CUDA_PATH%\lib\x64\cudart_static.lib" "%CUDA_PATH%\lib\x64\cublas.lib" /Fo:"%TEMP%\\" /Fe:"%3"
