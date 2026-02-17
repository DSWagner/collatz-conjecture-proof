@echo off
setlocal

set "INCLUDE="
set "LIB="
set "LIBPATH="
set "Path=C:\Windows\system32;C:\Windows;C:\Program Files\CMake\bin;C:\Program Files\Git\cmd;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;C:\Users\dswag\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe"

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
if errorlevel 1 (
    echo ERROR: vcvars64 failed
    exit /b 1
)

echo Compiler:
where cl.exe
cl.exe 2>&1 | findstr /i "version"
echo.

set PROJDIR=C:\Users\dswag\Desktop\PROJECTS\collatz_claude\collatz-conjecture-proof
cd /d "%PROJDIR%"

if exist build rmdir /s /q build
mkdir build
cd build

cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" -DCMAKE_CXX_COMPILER="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.40.33807\bin\Hostx64\x64\cl.exe" -DCMAKE_CUDA_HOST_COMPILER="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.40.33807\bin\Hostx64\x64\cl.exe"
if errorlevel 1 (
    echo ERROR: cmake configure failed
    exit /b 1
)

cmake --build . --parallel 16
if errorlevel 1 (
    echo ERROR: cmake build failed
    exit /b 1
)

echo.
echo BUILD SUCCESS
exit /b 0
