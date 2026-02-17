$ErrorActionPreference = 'Stop'
$proj   = 'C:\Users\dswag\Desktop\PROJECTS\collatz_claude\collatz-conjecture-proof'
$vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
$nvcc   = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe'
$cl     = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.40.33807\bin\Hostx64\x64\cl.exe'
$vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'

# Add vswhere to PATH so vcvars finds it
$env:PATH = "C:\Program Files (x86)\Microsoft Visual Studio\Installer;$env:PATH"

# Load vcvars64 into this PS session
Write-Host "[1/4] Loading VS 2022 environment..."
$envdump = & cmd /c "`"$vcvars`" && set" 2>&1
foreach ($line in $envdump) {
    if ($line -match '^([^=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
    }
}

Write-Host "     CL: $cl (VS 2022 14.40)"

# Configure
Write-Host "[2/4] Configuring CMake..."
if (Test-Path "$proj\build") { Remove-Item "$proj\build" -Recurse -Force }
New-Item "$proj\build" -ItemType Directory | Out-Null
Push-Location "$proj\build"

$cmakeOut = & cmake .. '-G' 'Ninja' '-DCMAKE_BUILD_TYPE=Release' `
    '-DCMAKE_CUDA_ARCHITECTURES=86' `
    "-DCMAKE_CUDA_COMPILER=$nvcc" `
    "-DCMAKE_CXX_COMPILER=$cl" `
    "-DCMAKE_CUDA_HOST_COMPILER=$cl" 2>&1
Write-Host ($cmakeOut -join "`n")
if ($LASTEXITCODE -ne 0) { Pop-Location; throw "CMake configure FAILED" }

# Build
Write-Host "[3/4] Building..."
$buildOut = & cmake --build . --parallel 16 2>&1
Write-Host ($buildOut -join "`n")
if ($LASTEXITCODE -ne 0) { Pop-Location; throw "Build FAILED" }

Pop-Location
Write-Host "[4/4] BUILD SUCCESS"
