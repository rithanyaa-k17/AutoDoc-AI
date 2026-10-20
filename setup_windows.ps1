# AutoDoc AI one-time setup (PowerShell, run from the project folder). Needs internet ONCE.
$ErrorActionPreference = "Stop"
if (-not (Test-Path .venv)) { python -m venv .venv }
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python Code\generate_synthetic.py
python Code\check_setup.py --download
Write-Host "Setup finished. Next: choose your LLM (README section 3), then: streamlit run Code\app.py"
