@rem Usage: build_harness.cmd <absolute-build-dir> <source.cpp in this folder> <output-exe>
@rem Links a backlog M3 CPU harness against the kan.lib of a Release build made by
@rem scripts\build.ps1 (CPU or CUDA), compiled with that tree's include directory.
@setlocal
@set ROOT=%~dp0..\..\..\..
@if not "%4"=="" set ROOT=%4
@call "C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /std:c++20 /O2 /EHsc /MD /I "%ROOT%\include" "%~dp0%2" "%1\kan.lib" /Fo:"%TEMP%\\" /Fe:"%3"
