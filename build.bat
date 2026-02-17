@echo off
:: =============================================================================
:: build.bat - Collatz Conjecture Proof Assistant
:: =============================================================================
:: Clears VS 2026 contamination, loads VS 2022, builds with Ninja + CMake.
:: =============================================================================

echo [BUILD] Clearing VS 2026 environment contamination...
set "INCLUDE="
set "LIB="
set "LIBPATH="
set "Path=C:\Windows\system32;C:\Windows;C:\Program Files\CMake\bin;C:\Program Files\Git\cmd;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;C:\Users\dswag\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe"

echo [BUILD] Loading VS 2022 environment...
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 (
    echo ERROR: Failed to load VS 2022 environment
    exit /b 1
)

echo [BUILD] Verifying compiler (must show 19.40.x, NOT 19.50.x)...
cl.exe 2>&1 | findstr /i "version"
where cl.exe

echo [BUILD] Configuring with CMake...
if exist build rmdir /s /q build
mkdir build
cd build

cmake .. -G "Ninja" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CUDA_ARCHITECTURES=86 ^
    -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" ^
    -DCMAKE_CXX_COMPILER="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.40.33807\bin\Hostx64\x64\cl.exe" ^
    -DCMAKE_CUDA_HOST_COMPILER="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.40.33807\bin\Hostx64\x64\cl.exe"

if errorlevel 1 (
    echo ERROR: CMake configuration failed
    cd ..
    exit /b 1
)

echo [BUILD] Building...
cmake --build . --parallel 16

if errorlevel 1 (
    echo ERROR: Build failed
    cd ..
    exit /b 1
)

cd ..
echo [BUILD] SUCCESS: collatz_proof.exe built in build\
echo.
echo Usage:
echo   build\collatz_proof.exe                          ^(all directions, defaults^)
echo   build\collatz_proof.exe --d2 100000000 --d3 1000000000
echo   build\collatz_proof.exe --only 1                 ^(D1 only: cycle analysis^)
echo   build\collatz_proof.exe --only 4                 ^(D4 only: drift bound^)
echo   build\collatz_proof.exe --only 12                ^(D1 + D2^)
echo   build\collatz_proof.exe --only 1234              ^(all^)
