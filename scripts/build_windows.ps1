$ErrorActionPreference = 'Stop'
$packageProject = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $packageProject

# PyInstaller discovers hooks with glob paths. Its installed location must not
# contain brackets, as this checkout does. An isolated build environment also
# keeps packaging dependencies out of the application's development environment.
$packageHasher = [Security.Cryptography.SHA256]::Create()
try {
    $packageHash = ([BitConverter]::ToString($packageHasher.ComputeHash(
        [Text.Encoding]::UTF8.GetBytes($packageProject)))).Replace('-', '').Substring(0, 12)
} finally {
    $packageHasher.Dispose()
}
$packageEnvironment = Join-Path $env:LOCALAPPDATA "HayDayAutomationBuild\$packageHash\venv"
$packagePython = Join-Path $packageEnvironment 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $packagePython)) {
    if (Test-Path -LiteralPath '.venv\Scripts\python.exe') {
        & '.venv\Scripts\python.exe' -m venv $packageEnvironment
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv $packageEnvironment
    } else {
        throw 'Run setup.ps1 first, or install Python 3.11 or newer to build the executable.'
    }
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the isolated build environment.' }
}
$packagePipArguments = @('-m', 'pip', 'install', '.[build]')
if (Test-Path -LiteralPath '.venv\Scripts\python.exe') {
    $packageConstraint = Join-Path (Split-Path -Parent $packageEnvironment) 'runtime-constraints.txt'
    & '.venv\Scripts\python.exe' -m pip freeze --exclude-editable | Set-Content -LiteralPath $packageConstraint -Encoding UTF8
    if ($LASTEXITCODE -ne 0) { throw 'Could not capture the tested dependency versions.' }
    $packagePipArguments += @('--constraint', $packageConstraint)
}
& $packagePython @packagePipArguments
if ($LASTEXITCODE -ne 0) { throw 'Could not install the build dependencies.' }
& $packagePython 'scripts\build_windows.py'
if ($LASTEXITCODE -ne 0) { throw 'The Windows executable build failed.' }
