$msvc  = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.40.33807\bin\Hostx64\x64'
$winsdk= 'C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64'
$ninja = 'C:\Users\dswag\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe'
$env:PATH    = "$msvc;$winsdk;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;$ninja;C:\Program Files\CMake\bin;C:\Windows\system32;C:\Windows"
$env:INCLUDE = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.40.33807\include;C:\Program Files (x86)\Windows Kits\10\include\10.0.26100.0\ucrt;C:\Program Files (x86)\Windows Kits\10\include\10.0.26100.0\um;C:\Program Files (x86)\Windows Kits\10\include\10.0.26100.0\shared'
$env:LIB     = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.40.33807\lib\x64;C:\Program Files (x86)\Windows Kits\10\lib\10.0.26100.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\lib\10.0.26100.0\um\x64'
$env:LIBPATH = $env:LIB

$cl   = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.40.33807\bin\Hostx64\x64\cl.exe'
$nvcc = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe'
$proj = 'C:\Users\dswag\Desktop\PROJECTS\collatz_claude\collatz-conjecture-proof'

if (Test-Path "$proj\build") { Remove-Item "$proj\build" -Recurse -Force }
New-Item "$proj\build" -ItemType Directory | Out-Null
Set-Location "$proj\build"

Write-Host "=== CMAKE CONFIGURE ===" -ForegroundColor Cyan
& cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 "-DCMAKE_CUDA_COMPILER=$nvcc" "-DCMAKE_CXX_COMPILER=$cl" "-DCMAKE_CUDA_HOST_COMPILER=$cl"
if ($LASTEXITCODE -ne 0) { Write-Error "Configure failed"; exit 1 }

Write-Host "=== CMAKE BUILD ===" -ForegroundColor Cyan
& cmake --build . --parallel 16
if ($LASTEXITCODE -ne 0) { Write-Error "Build failed"; exit 1 }

Write-Host "=== BUILD SUCCESS: $proj\build\collatz_proof.exe ===" -ForegroundColor Green
