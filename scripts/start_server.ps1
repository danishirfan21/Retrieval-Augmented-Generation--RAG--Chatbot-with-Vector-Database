<#
.SYNOPSIS
    Start the FastAPI server using the project's venv Python if present.

DESCRIPTION
    This small helper attempts to use `.\venv\Scripts\python.exe` if it exists,
    otherwise falls back to the system `python` on PATH.

USAGE
    .\scripts\start_server.ps1

#>

Write-Host "[INFO] Starting FastAPI server..." -ForegroundColor Cyan

if (Test-Path -Path ".\venv\Scripts\python.exe") {
    $python = Resolve-Path -Path ".\venv\Scripts\python.exe"
    $python = $python.Path
    Write-Host "[INFO] Using venv python: $python"
} else {
    $python = "python"
    Write-Host "[WARN] venv python not found; falling back to system 'python' on PATH"
}

$uvicornArgs = "-m uvicorn app.main:app --host 0.0.0.0 --port 8000"
Write-Host "[INFO] Launching: $python $uvicornArgs"

# Start server in background
if (-not (Test-Path -Path ".\logs")) { New-Item -ItemType Directory -Path .\logs | Out-Null }
Start-Process -FilePath $python -ArgumentList $uvicornArgs -NoNewWindow -RedirectStandardOutput .\logs\uvicorn_stdout.log -RedirectStandardError .\logs\uvicorn_stderr.log

Write-Host "[INFO] Server started in background. Logs: .\logs\uvicorn_*.log" -ForegroundColor Green
