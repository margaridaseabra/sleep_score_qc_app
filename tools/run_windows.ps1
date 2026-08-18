$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$AppEnvName = "sleep_stage_qc_v2"
$SomnotateCommit = "a20f33de62511d8c172e333896608b7fc166d0f0"
$SomnotateShortCommit = $SomnotateCommit.Substring(0, 8)
$SomnotateRoot = Join-Path $env:LOCALAPPDATA ("SleepStageQC\somnotate_" + $SomnotateShortCommit)
$Url = "http://127.0.0.1:8050"

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

    throw "Conda could not be found. Run INSTALL_WINDOWS.bat after installing Miniconda/Anaconda/Miniforge."
}

try {
    $CondaExe = Find-CondaExecutable

    if (-not (Test-Path (Join-Path $RepoRoot "dash_app\app.py"))) {
        throw "dash_app\app.py is missing. Re-download the Sleep Stage QC ZIP."
    }

    if (-not (Test-Path (Join-Path $SomnotateRoot "example_pipeline\01_preprocess_signals.py"))) {
        throw "The automatic Somnotate installation was not found. Run INSTALL_WINDOWS.bat first."
    }

    # The Dash app reads this and automatically fills the Somnotate repository field.
    $env:SOMNOTATE_ROOT = $SomnotateRoot

    Write-Host ""
    Write-Host "Sleep Stage QC" -ForegroundColor Green
    Write-Host "Repository: $RepoRoot"
    Write-Host "Somnotate: $SomnotateRoot"
    Write-Host "Opening: $Url"
    Write-Host ""
    Write-Host "Keep this window open while using the app."
    Write-Host "Press Ctrl+C here when you want to stop the app."
    Write-Host ""

    # Start the app without requiring the user to activate Conda manually.
    $arguments = @(
        "run", "--no-capture-output",
        "-n", $AppEnvName,
        "python", "-u", "-m", "dash_app.app"
    )

    $process = Start-Process `
        -FilePath $CondaExe `
        -ArgumentList $arguments `
        -WorkingDirectory $RepoRoot `
        -NoNewWindow `
        -PassThru

    # Wait until the Dash server responds, then open the default browser.
    $opened = $false
    for ($i = 0; $i -lt 60; $i++) {
        if ($process.HasExited) {
            break
        }

        try {
            $client = New-Object System.Net.Sockets.TcpClient
            $async = $client.BeginConnect("127.0.0.1", 8050, $null, $null)
            $connected = $async.AsyncWaitHandle.WaitOne(250, $false)
            if ($connected -and $client.Connected) {
                $client.EndConnect($async)
                $client.Close()
                Start-Process $Url
                $opened = $true
                break
            }
            $client.Close()
        }
        catch {
            # Server is simply not ready yet.
        }

        Start-Sleep -Milliseconds 500
    }

    if (-not $opened -and -not $process.HasExited) {
        Write-Host "The app is still starting. Open this address manually if needed:"
        Write-Host "  $Url"
    }

    $process.WaitForExit()

    if ($process.ExitCode -ne 0) {
        throw "The app exited with code $($process.ExitCode)."
    }

    exit 0
}
catch {
    Write-Host ""
    Write-Host "Could not start Sleep Stage QC:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Write-Host "If this is the first run, double-click INSTALL_WINDOWS.bat."
    exit 1
}
