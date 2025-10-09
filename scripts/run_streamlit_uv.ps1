param(
    [string]$AppPath = "technical_analysis_app.py"
)

# Ensure the script runs from repository root
Set-Location -Path (Resolve-Path "$PSScriptRoot\..")

if (-not (Test-Path ".venv")) {
    Write-Host "Creating project virtual environment with uv..." -ForegroundColor Cyan
    uv venv .venv
}

$originalLinkMode = $env:UV_LINK_MODE
$env:UV_LINK_MODE = "copy"

try {
    Write-Host "Syncing dependencies via uv..." -ForegroundColor Cyan
    uv sync

    Write-Host "Launching Streamlit with uv (without additional sync)..." -ForegroundColor Cyan
    uv run --no-sync streamlit run $AppPath
}
finally {
    if ($null -ne $originalLinkMode) {
        $env:UV_LINK_MODE = $originalLinkMode
    }
    else {
        Remove-Item Env:UV_LINK_MODE -ErrorAction SilentlyContinue
    }
}
