<#
.SYNOPSIS
Set up the Python project on Windows, including Python when needed.
.DESCRIPTION
Double-click Setup Hay Day.cmd. Use -NoLaunch to install without opening the
app, -Dev for development tools, or -Python to choose an existing interpreter.
#>
param([string]$Python = '', [switch]$Dev, [switch]$NoLaunch)

$ErrorActionPreference = 'Stop'

function Test-HayDayPython {
    param([string]$Executable, [switch]$VirtualEnvironment)
    if (-not $Executable -or -not (Test-Path -LiteralPath $Executable -PathType Leaf)) { return $false }
    # Do not invoke Windows Store execution aliases while looking for Python.
    if ($Executable -match '\\Microsoft\\WindowsApps\\') { return $false }
    $probe = "import sys,sysconfig,struct,platform,venv,ensurepip; valid = platform.python_implementation()=='CPython' and sys.version_info[:2] in ((3,12),(3,13)) and struct.calcsize('P')==8 and not sysconfig.get_config_var('Py_GIL_DISABLED')"
    if ($VirtualEnvironment) { $probe += '; valid = valid and sys.prefix != sys.base_prefix' }
    $probe += '; sys.exit(0 if valid else 1)'
    try {
        & $Executable -I -c $probe *> $null
        return ($LASTEXITCODE -eq 0)
    } catch { return $false }
}

function Find-HayDayPython {
    param([string]$Requested)
    if ($Requested) {
        if (-not (Test-HayDayPython $Requested)) {
            throw 'The selected Python must be a working standard 64-bit CPython 3.12 or 3.13 installation.'
        }
        return (Get-Item -LiteralPath $Requested).FullName
    }
    $candidates = @((Join-Path $env:LOCALAPPDATA 'Programs\HayDayAutomation\Python313\python.exe'))
    foreach ($version in @('3.13', '3.12')) {
        foreach ($registry in @('HKCU:\Software\Python\PythonCore', 'HKLM:\Software\Python\PythonCore')) {
            $key = Join-Path $registry ($version + '\InstallPath')
            if (Test-Path -LiteralPath $key) {
                $install = Get-Item -LiteralPath $key
                $candidates += $install.GetValue('ExecutablePath', '')
                $directory = $install.GetValue('', '')
                if ($directory) { $candidates += Join-Path $directory 'python.exe' }
            }
        }
    }
    foreach ($name in @('python.exe', 'python3.exe')) {
        $candidates += @(Get-Command $name -CommandType Application -All -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty Source)
    }
    foreach ($candidate in $candidates | Select-Object -Unique) {
        if (Test-HayDayPython $candidate) { return $candidate }
    }
    return $null
}

function Install-HayDayPython {
    # Official PSF full installer and SHA-256 from its release page:
    # https://www.python.org/downloads/release/python-31315/
    $version = '3.13.15'
    $digest = 'edec09c4853aeae9ac36efb8c9f95b6b8e2fee65eee56d9767a8b7c69c574403'
    $target = Join-Path $env:LOCALAPPDATA 'Programs\HayDayAutomation\Python313'
    $executable = Join-Path $target 'python.exe'
    if (Test-HayDayPython $executable) { return $executable }
    $cache = Join-Path $env:LOCALAPPDATA 'HayDayAutomationSetup'
    New-Item -ItemType Directory -Path $cache -Force | Out-Null
    $installer = Join-Path $cache "python-$version-amd64.exe"
    if (-not (Test-Path -LiteralPath $installer) -or
            (Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash -ne $digest) {
        Write-Host "Downloading Python $version from python.org..."
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        $oldProgress = $ProgressPreference
        try {
            $ProgressPreference = 'SilentlyContinue'
            Invoke-WebRequest -UseBasicParsing -TimeoutSec 180 -Uri "https://www.python.org/ftp/python/$version/python-$version-amd64.exe" -OutFile $installer
        } finally { $ProgressPreference = $oldProgress }
    }
    if ((Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash -ne $digest) {
        throw 'The Python download failed its integrity check. Run setup again to download a fresh copy.'
    }
    Write-Host "Installing Python $version for this Windows user..."
    $arguments = @('/quiet', '/norestart', 'InstallAllUsers=0', ('TargetDir="{0}"' -f $target),
        'Include_pip=1', 'Include_launcher=0', 'InstallLauncherAllUsers=0', 'Include_test=0',
        'Include_doc=0', 'Include_tcltk=0', 'Include_dev=0', 'Include_debug=0', 'Include_symbols=0',
        'Include_freethreaded=0', 'PrependPath=0', 'AppendPath=0', 'AssociateFiles=0', 'Shortcuts=0',
        '/log', ('"{0}"' -f (Join-Path $cache 'python-install.log')))
    $installProcess = Start-Process -FilePath $installer -ArgumentList $arguments -WorkingDirectory $env:TEMP -WindowStyle Hidden -Wait -PassThru
    if ($installProcess.ExitCode -notin @(0, 3010)) {
        throw "Python installation failed (exit $($installProcess.ExitCode)). See $cache\python-install.log."
    }
    if (-not (Test-HayDayPython $executable)) {
        throw "Python was not found after installation. Restart Windows if requested, then run setup again. See $cache\python-install.log."
    }
    return $executable
}

function Save-HayDayBrokenEnvironment {
    param([string]$ProjectRoot)
    $root = [IO.Path]::GetFullPath($ProjectRoot).TrimEnd('\')
    $environment = [IO.Path]::GetFullPath((Join-Path $root '.venv'))
    $backup = [IO.Path]::GetFullPath((Join-Path $root ('.venv.backup-' + [Guid]::NewGuid().ToString('N'))))
    if (-not (Test-Path -LiteralPath $environment)) { return }
    $prefix = $root + '\'
    if (-not $environment.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase) -or
            -not $backup.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'The environment backup must stay within the project.'
    }
    $item = Get-Item -LiteralPath $environment -Force
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        throw 'The .venv folder is a directory link. Choose a normal project folder before running setup.'
    }
    Write-Host ('Keeping the unusable environment as ' + [IO.Path]::GetFileName($backup))
    Move-Item -LiteralPath $environment -Destination $backup
}

function Invoke-HayDaySetup {
    param([string]$ProjectRoot = $PSScriptRoot, [string]$Python = '', [switch]$Dev, [switch]$NoLaunch)
    if (-not [Environment]::Is64BitOperatingSystem -or $env:OS -ne 'Windows_NT' -or
            $env:PROCESSOR_ARCHITEW6432 -eq 'ARM64' -or $env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {
        throw 'This setup requires Windows 10 or 11 on a 64-bit Intel/AMD computer.'
    }
    $root = [IO.Path]::GetFullPath($ProjectRoot)
    foreach ($required in @('app.py', 'pyproject.toml', 'requirements-windows.txt', 'scripts\check_setup.py', 'scripts\ui_smoke.py')) {
        if (-not (Test-Path -LiteralPath (Join-Path $root $required) -PathType Leaf)) {
            throw "Missing $required. Extract the whole Python project before running setup."
        }
    }
    $setupDirectory = Join-Path $root '.setup'
    New-Item -ItemType Directory -Path $setupDirectory -Force | Out-Null
    $lock = $null
    $transcript = $false
    Push-Location -LiteralPath $root
    try {
        try {
            $lock = [IO.File]::Open((Join-Path $setupDirectory 'setup.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
        } catch { throw 'Another setup is already running in this project. Let it finish first.' }
        Start-Transcript -LiteralPath (Join-Path $setupDirectory 'setup.log') -Force | Out-Null
        $transcript = $true
        Write-Host 'Setting up Hay Day Automation. Close the app before updating its environment.'
        $environment = Join-Path $root '.venv'
        $venvPython = Join-Path $environment 'Scripts\python.exe'
        if (-not (Test-HayDayPython $venvPython -VirtualEnvironment)) {
            $basePython = Find-HayDayPython $Python
            if (-not $basePython) { $basePython = Install-HayDayPython }
            Save-HayDayBrokenEnvironment $root
            Write-Host 'Creating the project Python environment...'
            & $basePython -I -m venv $environment
            if ($LASTEXITCODE -ne 0) { throw 'Could not create the project Python environment.' }
        }
        Write-Host 'Installing the tested libraries...'
        & $venvPython -I -m pip --disable-pip-version-check --no-input --log .setup/pip.log install --only-binary=:all: -r requirements-windows.txt
        if ($LASTEXITCODE -ne 0) { throw 'Library installation failed. Check your internet connection and run setup again.' }
        # Editable installation keeps the real .py files in use. Its build
        # backend is pinned above instead of downloaded in an isolated build.
        $project = if ($Dev) { '.[dev]' } else { '.' }
        & $venvPython -I -m pip --disable-pip-version-check --no-input --log .setup/pip.log install --no-build-isolation --constraint requirements-windows.txt --editable $project
        if ($LASTEXITCODE -ne 0) { throw 'The Python project could not be installed.' }
        & $venvPython -I -m pip check
        if ($LASTEXITCODE -ne 0) { throw 'Installed libraries have conflicting dependencies.' }
        Write-Host 'Preparing and checking the desktop interface and recognition libraries...'
        & $venvPython -I scripts/check_setup.py
        if ($LASTEXITCODE -ne 0) { throw 'The application dependency check failed.' }
        & $venvPython -I scripts/ui_smoke.py --hidden --seconds-per-route 0.2 --report .setup/ui-check.json
        if ($LASTEXITCODE -ne 0) { throw 'The desktop interface check failed. See .setup\ui-check.json.' }
        Write-Host 'Setup complete. Use Launch Hay Day.cmd to open the app.'
        Write-Host 'BlueStacks or MuMu Player with Hay Day and ADB enabled is required for game automation.'
        if (-not $NoLaunch) {
            Start-Process -FilePath (Join-Path $environment 'Scripts\pythonw.exe') -ArgumentList ('-I "{0}"' -f (Join-Path $root 'app.py')) -WorkingDirectory $env:TEMP -WindowStyle Hidden | Out-Null
        }
    } finally {
        if ($transcript) { Stop-Transcript | Out-Null }
        if ($lock) { $lock.Dispose() }
        Pop-Location
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    try { Invoke-HayDaySetup -Python $Python -Dev:$Dev -NoLaunch:$NoLaunch }
    catch {
        Write-Host ('Setup failed: ' + $_.Exception.Message) -ForegroundColor Red
        Write-Host 'See .setup\setup.log for details. Run Setup Hay Day.cmd again after correcting the problem.'
        exit 1
    }
}
