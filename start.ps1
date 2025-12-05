# Sustainable AI - Energy Efficient Prompt Engineering
# Windows PowerShell Startup Script

Write-Host "========================================"
Write-Host "  Sustainable AI Energy Monitor"
Write-Host "  Starting application..."
Write-Host "========================================"
Write-Host ""

# Change to project directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Check if virtual environment exists
$VenvPath = Join-Path $ScriptDir "venv\Scripts\Activate.ps1"
if (Test-Path $VenvPath) {
    Write-Host "Activating virtual environment..."
    & $VenvPath
}
else {
    Write-Host "No virtual environment found. Using system Python."
}

# Check if streamlit is installed
$StreamlitCheck = python -c "import streamlit" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing required packages..."
    pip install -r requirements.txt
}

Write-Host ""
Write-Host "Starting Streamlit application..."
Write-Host ""
Write-Host "Access the application at: http://localhost:8501"
Write-Host "Press Ctrl+C to stop the server."
Write-Host ""

# Run the Streamlit application
streamlit run src\gui\app.py --server.port 8501 --server.address localhost
