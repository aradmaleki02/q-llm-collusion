param([switch]$Isolated)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw 'Install Python 3.11 or newer, with Python added to PATH, then rerun setup.ps1.'
}
python -c "import sys; assert sys.version_info >= (3,11), 'Python 3.11+ is required'"
if ($LASTEXITCODE -ne 0) { throw 'Unsupported Python version.' }
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    if ($Isolated) { python -m venv .venv }
    else { python -m venv --system-site-packages .venv }
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the virtual environment.' }
}
$experimentPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $experimentPython -c "import torch; assert (2,5) <= tuple(map(int,torch.__version__.split('+')[0].split('.')[:2])) < (3,0)"
if ($LASTEXITCODE -ne 0) {
    & $experimentPython -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.5,<3'
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed. Check the network and rerun setup.ps1.' }
}
if (-not (Test-Path -LiteralPath '.env')) {
    Copy-Item -LiteralPath '.env.example' -Destination '.env'
}
& $experimentPython -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'Local verification failed.' }
Write-Host 'Setup complete. Edit .env locally to add OPENAI_API_KEY.'
Write-Host 'Offline check: .\.venv\Scripts\python.exe -m collusion run --name smoke --backend mock --rounds 12 --train-episodes 1 --eval-episodes 1'
Write-Host 'Live experiment: .\.venv\Scripts\python.exe -m collusion run --name pilot'
