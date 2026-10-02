@rem Usage: build_profile.cmd <build-dir> <output-exe> [include-dir]
@rem Links rational_policy_profile.cpp against a Release CUDA build made by scripts\build.ps1 -Cuda.
@rem For a pre-M2 baseline pass that commit's include directory (for example
@rem extracted with git archive); the harness then builds with /DKAN_NO_POLICY.
@setlocal
@set ROOT=%~dp0..\..\..
@set INC=%ROOT%\include
@set EXTRA=
@if not "%~3"=="" (set "INC=%~3" & set EXTRA=/DKAN_NO_POLICY)
@call "C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /std:c++20 /O2 /EHsc /MD %EXTRA% /I "%INC%" "%~dp0rational_policy_profile.cpp" "%ROOT%\%1\kan_cuda.lib" "%ROOT%\%1\kan.lib" "%CUDA_PATH%\lib\x64\cudart_static.lib" "%CUDA_PATH%\lib\x64\cublas.lib" /Fo:"%TEMP%\\" /Fe:"%2"
