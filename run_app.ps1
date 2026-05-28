# ASD Screening Prediction System - Startup Script
# Right-click and select "Run with PowerShell"

Write-Host "========================================"
Write-Host "ASD Screening Prediction System"
Write-Host "========================================"
Write-Host ""

# Check if Python is installed
Write-Host "Checking Python installation..."
$python = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Python found: $python" -ForegroundColor Green
} else {
    Write-Host "✗ Python not found. Please install Python 3.8 or higher" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit
}

Write-Host ""
Write-Host "Installing/Updating dependencies..."
Write-Host "This may take a few minutes on first run..."
Write-Host ""

pip install -r requirements.txt --upgrade

Write-Host ""
Write-Host "Starting Streamlit application..." -ForegroundColor Green
Write-Host ""
Write-Host "The app will open in your browser at:" -ForegroundColor Cyan
Write-Host "http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

Start-Sleep -Seconds 2

streamlit run streamlit_app.py --server.fileWatcherType none
