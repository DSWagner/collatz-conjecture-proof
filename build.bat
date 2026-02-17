@echo off
echo ================================================================
echo    Collatz Conjecture Proof Assistant v1.0.0 - Build
echo ================================================================

if not exist build mkdir build
cd build

cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 ^
    -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" ^
    -DCMAKE_CXX_COMPILER="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.40.33807\bin\Hostx64\x64\cl.exe" ^
    -DCMAKE_CUDA_HOST_COMPILER="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.40.33807\bin\Hostx64\x64\cl.exe"

cmake --build . --parallel 16

echo.
echo ================================================================
echo Build complete!
echo.
echo Usage:
echo   build\collatz_proof.exe
echo   build\collatz_proof.exe --d2 100000000 --d3 1000000000
echo   build\collatz_proof.exe --only 4   (D4 drift only, ~1 sec)
echo ================================================================
