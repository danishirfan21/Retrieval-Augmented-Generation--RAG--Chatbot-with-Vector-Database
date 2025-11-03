<#
.SYNOPSIS
    Setup helper for the RAG chatbot project (Windows PowerShell).

DESCRIPTION
    Automates common local setup tasks:
      - checks python version
      - creates a venv at .\venv
      - installs dependencies from requirements.txt
      - copies .env.example to .env if missing
      - optionally runs document ingestion and/or starts the API server

USAGE
    # Basic install and copy .env example
    .\scripts\setup.ps1

    # Install + ingest documents + start server
    .\scripts\setup.ps1 -Ingest -RunServer

    # Skip pip install (assumes you installed manually)
    .\scripts\setup.ps1 -SkipInstall

PARAMETER SkipInstall
    Skip installing Python packages.

PARAMETER Ingest
    Run the ingestion script after installing dependencies.

PARAMETER RunServer
    Start the FastAPI server after installing dependencies.

PARAMETER UseDocker
    Use Docker Compose to run the app instead of a local venv-based server.

#>

[CmdletBinding()]
param(
    [switch]$SkipInstall,
    [switch]$Ingest,
    [switch]$RunServer,
    [switch]$UseDocker
)

function Write-Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host "[ERROR] $msg" -ForegroundColor Red }

Write-Info "Starting project setup..."

# 1) Check Python
try {
    $py = & python --version 2>&1
    if ($LASTEXITCODE -ne 0) { throw "python not found on PATH" }
    Write-Info "Detected Python: $py"

    # Parse version number
    $verText = ($py -split ' ')[1]
    $parts = $verText -split '\.' | ForEach-Object {[int]$_}
    if ($parts[0] -lt 3 -or ($parts[0] -eq 3 -and $parts[1] -lt 10)) {
        Write-Warn "Python 3.10+ is recommended. Detected $verText - proceed at your own risk."
    }
} catch {
    Write-Err "Python not found. Please install Python 3.10+ and ensure 'python' is on PATH."
    exit 1
}

# 2) Copy .env.example -> .env if missing
if (-not (Test-Path -Path ".\.env")) {
    if (Test-Path -Path ".\.env.example") {
    Copy-Item -Path ".\.env.example" -Destination ".\.env"
    Write-Info "Copied .env.example to .env - open .env and add your API keys (OPENAI/PINECONE)."
    } else {
    Write-Warn ".env.example not found - create a .env file with required keys (OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME)."
    }
} else {
    Write-Info ".env already exists - leaving it unchanged."
}

# 3) Create venv
if (-not (Test-Path -Path ".\venv\Scripts\python.exe")) {
    Write-Info "Creating virtual environment at .\venv..."
    & python -m venv .\venv
    if ($LASTEXITCODE -ne 0) { Write-Warn "Failed to create venv; you may need permissions or to run PowerShell as admin." }
} else {
    Write-Info "Virtual environment already exists at .\venv"
}

if (Test-Path -Path ".\venv\Scripts\python.exe") {
    $venvPy = (Resolve-Path ".\venv\Scripts\python.exe").Path
} else {
    # Fall back to system python if venv python not available
    $venvPy = "python"
}

# 4) Install requirements
if (-not $SkipInstall) {
    Write-Info "Upgrading pip and installing requirements... (this can take a while)"
    & $venvPy -m pip install --upgrade pip 2>&1 | Write-Host
    if (Test-Path -Path ".\requirements.txt") {
        & $venvPy -m pip install -r .\requirements.txt 2>&1 | Write-Host
        if ($LASTEXITCODE -ne 0) { Write-Warn "Some packages may have failed to install. Check the output above and install dependencies manually if needed." }
    } else {
        Write-Warn "requirements.txt not found in repository root. Skipping pip install."
    }
} else {
    Write-Info "Skipping pip install as requested."
}

# 5) Optional: Ingest documents
if ($Ingest) {
    Write-Info "Running document ingestion..."
    & $venvPy .\scripts\ingest_documents.py 2>&1 | Write-Host
    if ($LASTEXITCODE -ne 0) { Write-Warn "Ingestion script returned non-zero exit code; check logs above." }
}

# 6) Optional: start server (or use Docker)
if ($RunServer -and -not $UseDocker) {
    Write-Info "Starting FastAPI server with uvicorn (foreground)..."
    Write-Info "Use Ctrl+C to stop the server."
    & $venvPy -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
}

if ($UseDocker) {
    if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
        Write-Warn "docker-compose not found on PATH. Install Docker Desktop and ensure docker-compose is available."
    } else {
        Write-Info "Starting containers with docker-compose..."
        Start-Process -NoNewWindow -FilePath docker-compose -ArgumentList 'up','--build'
    }
}

Write-Info "Setup script completed. Next steps: edit .env with keys, run ingestion (if not run), then start the server or use Docker as needed."
