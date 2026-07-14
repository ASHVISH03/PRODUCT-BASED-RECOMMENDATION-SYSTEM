# start.ps1 — Windows PowerShell server start script
# Run from the project root: .\scripts\start.ps1

param(
    [int]$Port = 8000,
    [switch]$Production,
    [switch]$MLflow
)

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Product Recommendation System — Server" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Activate virtual environment
if (Test-Path ".\venv\Scripts\Activate.ps1") {
    Write-Host "`nActivating virtual environment..." -ForegroundColor Yellow
    & ".\venv\Scripts\Activate.ps1"
}

# Check model exists
if (-not (Test-Path "models\tfidf.pkl")) {
    Write-Host "`n[WARNING] Model artifacts not found." -ForegroundColor Yellow
    Write-Host "  The API will start but recommendations won't work until training is run." -ForegroundColor Yellow
    Write-Host "  Train first: .\scripts\train.ps1`n" -ForegroundColor Cyan
}

# Optionally start MLflow in background
if ($MLflow) {
    Write-Host "Starting MLflow UI on http://localhost:5000..." -ForegroundColor Yellow
    Start-Process -FilePath "python" -ArgumentList "-m mlflow ui --host 0.0.0.0 --port 5000" -NoNewWindow
}

# Build uvicorn command
$workers = if ($Production) { 4 } else { 1 }
$reload = if ($Production) { "" } else { "--reload" }

Write-Host "`nStarting FastAPI server..." -ForegroundColor Green
Write-Host "  URL:     http://localhost:$Port" -ForegroundColor Cyan
Write-Host "  Docs:    http://localhost:$Port/docs" -ForegroundColor Cyan
Write-Host "  Admin:   http://localhost:$Port/admin.html" -ForegroundColor Cyan
if ($MLflow) {
    Write-Host "  MLflow:  http://localhost:5000" -ForegroundColor Cyan
}
Write-Host "  Mode:    $(if ($Production) { 'Production' } else { 'Development (--reload)' })" -ForegroundColor Cyan
Write-Host "`nPress Ctrl+C to stop`n" -ForegroundColor Yellow

if ($Production) {
    uvicorn app.main:app --host 0.0.0.0 --port $Port --workers $workers
} else {
    uvicorn app.main:app --host 0.0.0.0 --port $Port --reload
}
