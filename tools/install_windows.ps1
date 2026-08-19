$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$AppEnvName = "sleep_stage_qc_v2"
$SomEnvName = "somnotate_env"

# Somnotate source revision validated with Sleep Stage QC v1.1.
$SomnotateCommit = "a20f33de62511d8c172e333896608b7fc166d0f0"
$SomnotateShortCommit = $SomnotateCommit.Substring(0, 8)

$AppEnvFile = Join-Path $RepoRoot "environment.yml"
$SomEnvFile = Join-Path $RepoRoot "environment_somnotate.yml"
$CheckSetup = Join-Path $RepoRoot "check_setup.py"
$SourceContractCheck = Join-Path $RepoRoot "tools\check_somnotate_source_contract.py"

$LocalRoot = Join-Path $env:LOCALAPPDATA "SleepStageQC"
$SomnotateRoot = Join-Path $LocalRoot ("somnotate_" + $SomnotateShortCommit)
$InstallInfo = Join-Path $LocalRoot "windows_install_info.txt"

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Find-CondaExecutable {
    $cmd = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) {
        return $cmd.Source
    }

    $candidates = @(
        (Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"),
        (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"),
        (Join-Path $env:USERPROFILE "miniforge3\Scripts\conda.exe"),
        (Join-Path $env:USERPROFILE "mambaforge\Scripts\conda.exe"),
        (Join-Path $env:LOCALAPPDATA "miniconda3\Scripts\conda.exe"),
        (Join-Path $env:LOCALAPPDATA "anaconda3\Scripts\conda.exe"),
        (Join-Path $env:LOCALAPPDATA "miniforge3\Scripts\conda.exe"),
        (Join-Path $env:ProgramData "miniconda3\Scripts\conda.exe"),
        (Join-Path $env:ProgramData "anaconda3\Scripts\conda.exe")
    )

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return (Resolve-Path $candidate).Path
        }
    }

    throw @"
Conda could not be found.

Install Miniconda, Anaconda, or Miniforge first, then run INSTALL_WINDOWS.bat again.
You do NOT need to create any environments manually.
"@
}

function Invoke-Conda {
    param(
        [Parameter(Mandatory=$true)][string[]]$Arguments,
        [Parameter(Mandatory=$true)][string]$Description
    )

    Write-Step $Description
    & $script:CondaExe @Arguments
    $code = $LASTEXITCODE

    if ($code -ne 0) {
        throw "$Description failed (exit code $code)."
    }
}

function Get-CondaEnvironments {
    $raw = & $script:CondaExe env list --json
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read the Conda environment list."
    }

    return ($raw | Out-String | ConvertFrom-Json).envs
}

function Get-EnvironmentPrefix([string]$Name) {
    foreach ($prefixText in (Get-CondaEnvironments)) {
        if ($prefixText -and (Split-Path $prefixText -Leaf) -eq $Name) {
            return [string]$prefixText
        }
    }
    return $null
}

function Get-EnvironmentPython([string]$Name) {
    $prefix = Get-EnvironmentPrefix $Name
    if (-not $prefix) {
        return $null
    }

    $python = Join-Path $prefix "python.exe"
    if (-not (Test-Path $python)) {
        return $null
    }

    return $python
}

function Get-FreeSpaceGB([string]$Path) {
    $full = [System.IO.Path]::GetFullPath($Path)
    $root = [System.IO.Path]::GetPathRoot($full)
    $drive = New-Object System.IO.DriveInfo($root)
    return [math]::Round($drive.AvailableFreeSpace / 1GB, 2)
}

function Assert-FreeSpaceForSolve {
    $condaRoot = Split-Path (Split-Path $script:CondaExe -Parent) -Parent
    $freeGB = Get-FreeSpaceGB $condaRoot
    Write-Host ("Free disk space on Conda drive: {0} GB" -f $freeGB)

    if ($freeGB -lt 8) {
        throw @"
Not enough free disk space to create or repair a Conda environment.

Available: $freeGB GB
Required before environment creation/repair: at least 8 GB

Free some space and run INSTALL_WINDOWS.bat again.

A safe Conda cache cleanup command is:
  conda clean --all --yes

This removes downloaded/package caches, not your named Conda environments.
For a comfortable fresh installation, about 10 GB free is recommended.
"@
    }
}

function Invoke-PythonHealthCheck {
    param(
        [Parameter(Mandatory=$true)][string]$PythonExe,
        [Parameter(Mandatory=$true)][string]$Code,
        [Parameter(Mandatory=$true)][string]$Label
    )

    $checkDir = Join-Path $env:TEMP "SleepStageQC_installer_checks"
    New-Item -ItemType Directory -Path $checkDir -Force | Out-Null
    $checkFile = Join-Path $checkDir (($Label -replace '[^A-Za-z0-9_-]', '_') + ".py")
    Set-Content -Path $checkFile -Value $Code -Encoding UTF8

    try {
        & $PythonExe $checkFile
        return ($LASTEXITCODE -eq 0)
    }
    finally {
        Remove-Item $checkFile -Force -ErrorAction SilentlyContinue
    }
}

function Test-AppEnvironment {
    Write-Step "Checking existing app environment '$AppEnvName'"

    $python = Get-EnvironmentPython $AppEnvName
    if (-not $python) {
        return $false
    }

    $code = @'
import sys
try:
    import numpy
    import pandas
    import scipy
    import sklearn
    import dash
    import plotly
    import h5py
    import pyedflib
except Exception as exc:
    print(f"App environment check failed: {exc}")
    raise SystemExit(1)

if sys.version_info[:2] != (3, 11):
    print(f"Expected Python 3.11, found {sys.version.split()[0]}")
    raise SystemExit(1)

print("Existing app environment is usable.")
'@

    if (-not (Invoke-PythonHealthCheck -PythonExe $python -Code $code -Label "app_env_check")) {
        return $false
    }

    $prefix = Split-Path $python -Parent
    $ffmpeg = Join-Path $prefix "Library\bin\ffmpeg.exe"
    if (-not (Test-Path $ffmpeg)) {
        Write-Host "FFmpeg is missing from the existing app environment."
        return $false
    }

    return $true
}

function Test-SomnotateEnvironment {
    Write-Step "Checking existing Somnotate environment '$SomEnvName'"

    $python = Get-EnvironmentPython $SomEnvName
    if (-not $python) {
        return $false
    }

    $code = @'
import sys
try:
    import numpy
    import pandas
    import scipy
    import sklearn
    import matplotlib
    import pyedflib
    import lspopt
    import pomegranate
except Exception as exc:
    print(f"Somnotate environment check failed: {exc}")
    raise SystemExit(1)

if sys.version_info[:2] != (3, 9):
    print(f"Expected Python 3.9, found {sys.version.split()[0]}")
    raise SystemExit(1)

expected = {
    "numpy": "1.26.4",
    "pandas": "2.3.3",
    "scikit-learn": "1.6.1",
    "scipy": "1.13.1",
    "matplotlib": "3.9.4",
    "pyedflib": "0.1.42",
    "lspopt": "1.4.0",
    "pomegranate": "0.14.4",
}

found = {
    "numpy": numpy.__version__,
    "pandas": pandas.__version__,
    "scikit-learn": sklearn.__version__,
    "scipy": scipy.__version__,
    "matplotlib": matplotlib.__version__,
    "pyedflib": pyedflib.__version__,
    "lspopt": lspopt.__version__,
    "pomegranate": pomegranate.__version__,
}

bad = [
    f"{name}: found {found[name]}, expected {wanted}"
    for name, wanted in expected.items()
    if str(found[name]) != wanted
]

if bad:
    print("Somnotate environment versions do not match:")
    for item in bad:
        print("  " + item)
    raise SystemExit(1)

print("Existing Somnotate environment is usable.")
'@

    return (Invoke-PythonHealthCheck -PythonExe $python -Code $code -Label "somnotate_env_check")
}

function Ensure-AppEnvironment {
    $prefix = Get-EnvironmentPrefix $AppEnvName

    if ($prefix) {
        if (Test-AppEnvironment) {
            Write-Host "Skipping Conda solve because the existing app environment is healthy." -ForegroundColor Green
            return
        }

        Write-Host "The existing app environment needs repair."
        Assert-FreeSpaceForSolve
        Invoke-Conda `
            -Arguments @("env", "update", "-n", $AppEnvName, "-f", $AppEnvFile, "--prune", "--solver", "classic") `
            -Description "Repairing app environment '$AppEnvName' with Conda classic solver"
    }
    else {
        Assert-FreeSpaceForSolve
        Invoke-Conda `
            -Arguments @("env", "create", "-n", $AppEnvName, "-f", $AppEnvFile, "--solver", "classic") `
            -Description "Creating app environment '$AppEnvName' with Conda classic solver"
    }
}

function Ensure-SomnotateEnvironment {
    $prefix = Get-EnvironmentPrefix $SomEnvName

    if ($prefix) {
        if (Test-SomnotateEnvironment) {
            Write-Host "Skipping Conda solve because the existing Somnotate environment is healthy." -ForegroundColor Green
            return
        }

        Write-Host "The existing Somnotate environment needs repair."
        Assert-FreeSpaceForSolve
        Invoke-Conda `
            -Arguments @("env", "update", "-n", $SomEnvName, "-f", $SomEnvFile, "--prune", "--solver", "classic") `
            -Description "Repairing Somnotate environment '$SomEnvName' with Conda classic solver"
    }
    else {
        Assert-FreeSpaceForSolve
        Invoke-Conda `
            -Arguments @("env", "create", "-n", $SomEnvName, "-f", $SomEnvFile, "--solver", "classic") `
            -Description "Creating Somnotate environment '$SomEnvName' with Conda classic solver"
    }
}

function Ensure-SomnotateSource {
    $requiredPipeline = Join-Path $SomnotateRoot "example_pipeline\01_preprocess_signals.py"
    $marker = Join-Path $SomnotateRoot ".sleep_stage_qc_commit"

    if ((Test-Path $requiredPipeline) -and (Test-Path $marker)) {
        $installedCommit = (Get-Content $marker -Raw).Trim()
        if ($installedCommit -eq $SomnotateCommit) {
            Write-Step "Somnotate source already installed"
            Write-Host "Using: $SomnotateRoot"
            return
        }
    }

    Write-Step "Downloading supported Somnotate source from official GitHub"

    New-Item -ItemType Directory -Path $LocalRoot -Force | Out-Null

    $tempRoot = Join-Path $LocalRoot ("somnotate_download_" + [Guid]::NewGuid().ToString("N"))
    $zipPath = Join-Path $tempRoot "somnotate.zip"
    $extractPath = Join-Path $tempRoot "extract"
    New-Item -ItemType Directory -Path $extractPath -Force | Out-Null

    try {
        $url = "https://github.com/paulbrodersen/somnotate/archive/$SomnotateCommit.zip"
        Write-Host "Source: $url"
        Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing

        Expand-Archive -Path $zipPath -DestinationPath $extractPath -Force

        $downloaded = Join-Path $extractPath ("somnotate-" + $SomnotateCommit)
        if (-not (Test-Path $downloaded)) {
            $downloaded = Get-ChildItem -Path $extractPath -Directory |
                Select-Object -First 1 -ExpandProperty FullName
        }

        if (-not $downloaded -or -not (Test-Path $downloaded)) {
            throw "Somnotate downloaded, but the extracted source folder could not be found."
        }

        if (Test-Path $SomnotateRoot) {
            Remove-Item -Path $SomnotateRoot -Recurse -Force
        }

        Move-Item -Path $downloaded -Destination $SomnotateRoot
        Set-Content `
            -Path (Join-Path $SomnotateRoot ".sleep_stage_qc_commit") `
            -Value $SomnotateCommit `
            -Encoding ASCII

        if (-not (Test-Path $requiredPipeline)) {
            throw "The downloaded Somnotate source does not contain the expected example pipeline."
        }

        Write-Host "Installed Somnotate source at:"
        Write-Host "  $SomnotateRoot"
    }
    finally {
        if (Test-Path $tempRoot) {
            Remove-Item -Path $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

try {
    Write-Host ""
    Write-Host "Sleep Stage QC - automatic Windows setup" -ForegroundColor Green
    Write-Host "Repository: $RepoRoot"

    foreach ($required in @($AppEnvFile, $SomEnvFile, $CheckSetup, $SourceContractCheck)) {
        if (-not (Test-Path $required)) {
            throw "Required repository file is missing: $required"
        }
    }

    $script:CondaExe = Find-CondaExecutable
    $condaDir = Split-Path $script:CondaExe -Parent
    $env:PATH = "$condaDir;$env:PATH"

    # Use the classic solver for any environment creation/repair operation.
    $env:CONDA_SOLVER = "classic"

    Write-Step "Conda detected"
    Write-Host $script:CondaExe
    & $script:CondaExe --version
    if ($LASTEXITCODE -ne 0) {
        throw "Conda was found but could not run."
    }

    Ensure-AppEnvironment
    Ensure-SomnotateEnvironment
    Ensure-SomnotateSource

    $appPython = Get-EnvironmentPython $AppEnvName
    $somPython = Get-EnvironmentPython $SomEnvName

    if (-not $appPython) {
        throw "Could not locate Python for '$AppEnvName' after installation."
    }
    if (-not $somPython) {
        throw "Could not locate Python for '$SomEnvName' after installation."
    }

    Write-Step "Checking Somnotate source compatibility"
    & $appPython $SourceContractCheck $SomnotateRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Somnotate source compatibility check failed (exit code $LASTEXITCODE)."
    }

    Write-Step "Installing Somnotate into '$SomEnvName'"
    & $somPython -m pip install -e $SomnotateRoot --no-deps
    if ($LASTEXITCODE -ne 0) {
        throw "Installing Somnotate failed (exit code $LASTEXITCODE)."
    }

    $env:SOMNOTATE_ROOT = $SomnotateRoot
    $env:CONDA_DEFAULT_ENV = $AppEnvName

    # We run check_setup.py directly with the app environment's Python instead
    # of activating Conda. Add the same Windows runtime directories Conda would
    # normally place on PATH so tools such as ffmpeg are discoverable.
    $appPrefix = Split-Path $appPython -Parent
    $appPathParts = @(
        $appPrefix,
        (Join-Path $appPrefix "Scripts"),
        (Join-Path $appPrefix "Library\bin"),
        (Join-Path $appPrefix "Library\usr\bin"),
        (Join-Path $appPrefix "Library\mingw-w64\bin")
    ) | Where-Object { Test-Path $_ }
    $env:PATH = (($appPathParts -join ";") + ";" + $env:PATH)

    Write-Step "Running final setup diagnostics"
    & $appPython $CheckSetup `
        --somnotate-root $SomnotateRoot `
        --somnotate-env $SomEnvName `
        --require-somnotate

    if ($LASTEXITCODE -ne 0) {
        throw "Final setup diagnostics failed (exit code $LASTEXITCODE)."
    }

    New-Item -ItemType Directory -Path $LocalRoot -Force | Out-Null

    @"
Sleep Stage QC Windows installation

App repository:
$RepoRoot

App Conda environment:
$AppEnvName

Somnotate Conda environment:
$SomEnvName

Somnotate source:
$SomnotateRoot

Somnotate supported commit:
$SomnotateCommit

Installed:
$(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
"@ | Set-Content -Path $InstallInfo -Encoding UTF8

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "Installation successful." -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Somnotate is installed automatically at:"
    Write-Host "  $SomnotateRoot"
    Write-Host ""
    Write-Host "You do not need to enter the Somnotate path manually."
    Write-Host "Double-click RUN_APP.bat to start Sleep Stage QC."
    exit 0
}
catch {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host "INSTALLATION FAILED" -ForegroundColor Red
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Write-Host "The installer can be run again after the problem is fixed."
    exit 1
}
