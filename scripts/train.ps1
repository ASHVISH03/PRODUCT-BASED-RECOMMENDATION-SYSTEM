# train.ps1 — Windows PowerShell training script
# Run from the project root: .\scripts\train.ps1

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Product Recommendation System — Training" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Activate virtual environment if it exists
if (Test-Path ".\venv\Scripts\Activate.ps1") {
    Write-Host "`nActivating virtual environment..." -ForegroundColor Yellow
    & ".\venv\Scripts\Activate.ps1"
} else {
    Write-Host "`nNo venv found — using system Python" -ForegroundColor Yellow
}

# Check dataset exists
if (-not (Test-Path "data\raw\amazon.csv")) {
    Write-Host "`n[ERROR] Dataset not found: data\raw\amazon.csv" -ForegroundColor Red
    Write-Host "Run first: python scripts\setup_dataset.py" -ForegroundColor Yellow

    $answer = Read-Host "`nUse development fallback instead? [y/N]"
    if ($answer -eq "y") {
        Write-Host "Generating fallback dataset..." -ForegroundColor Yellow
        python scripts\generate_fallback.py
        Write-Host "Setting use_fallback: true in config..." -ForegroundColor Yellow
        (Get-Content src\config\config.yaml) -replace "use_fallback: false", "use_fallback: true" |
            Set-Content src\config\config.yaml
    } else {
        Write-Host "Training cancelled." -ForegroundColor Red
        exit 1
    }
}

# Run training pipeline
Write-Host "`nStarting training pipeline..." -ForegroundColor Green
$startTime = Get-Date

python -m src.pipelines.training_pipeline

$exitCode = $LASTEXITCODE
$duration = (Get-Date) - $startTime

if ($exitCode -eq 0) {
    Write-Host "`n✅ Training completed in $($duration.TotalSeconds.ToString('F1'))s" -ForegroundColor Green
    Write-Host "   Models saved to: models\" -ForegroundColor Cyan
    Write-Host "   MLflow runs: mlruns\" -ForegroundColor Cyan
    Write-Host "`nNext step: .\scripts\start.ps1" -ForegroundColor Cyan
} else {
    Write-Host "`n❌ Training failed (exit code: $exitCode)" -ForegroundColor Red
    Write-Host "   Check logs\training.log for details" -ForegroundColor Yellow
    exit $exitCode
}
