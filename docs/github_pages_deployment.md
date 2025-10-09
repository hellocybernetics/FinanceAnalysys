# Deploying the Technical Analysis UI to GitHub Pages

This guide walks through publishing the Streamlit-based technical analysis UI to GitHub Pages as a static site. The workflow leverages the `streamlit static export` command introduced in Streamlit 1.32 to produce static assets that GitHub Pages can host.

## Prerequisites

- Python 3.10 or later (matching the version defined in `pyproject.toml`).
- A fork or clone of the `FinanceAnalysys` repository with push access.
- GitHub Pages enabled for the repository (requires repository admin rights).
- Node.js is **not** required; the build uses Streamlit's static exporter.

## 1. Set up the environment

```bash
# Clone the repository if you have not already
git clone https://github.com/<your-account>/FinanceAnalysys.git
cd FinanceAnalysys

# Option A: Create a virtual environment with venv
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
pip install --upgrade pip
pip install -r requirements.txt

# Option B: Use uv for a faster, parallelized setup
uv venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
uv sync
```

> 💡 `uv sync` reads the dependency graph from `pyproject.toml` and installs packages in parallel, which is considerably faster than sequential `pip install` runs. The repository also exposes a `poetry` configuration if you prefer Poetry over `pip` or `uv`.

## 2. Verify the application locally

Before exporting the site, confirm the UI behaves as expected:

```bash
# Traditional activation flow
streamlit run technical_analysis_app.py

# Using uv directly without activating the virtual environment
uv run streamlit run technical_analysis_app.py
```

> ⚠️ **Windows + uv tip:** If you see `アクセスが拒否されました (.venv\lib64)` while running the uv command, Windows is
> blocking uv from recreating the `lib64` junction inside the project virtual environment. Run `set UV_LINK_MODE=copy`
> in the same terminal (or use `scripts\run_streamlit_uv.ps1`, described below) before invoking `uv sync`/`uv run` to
> force uv to copy files instead of creating symlinks.

Open the provided local URL in your browser, exercise the single-symbol, multi-symbol, and backtesting flows, then stop the server (Ctrl+C) when finished.

### Windows helper script

The repository ships with `scripts/run_streamlit_uv.ps1`, which wraps the uv commands with the `UV_LINK_MODE=copy` workaround and
skips redundant dependency syncs on every launch. From PowerShell run:

```powershell
pwsh -File scripts/run_streamlit_uv.ps1
```

You can pass a different entry point if needed (for example `-AppPath scripts/custom_app.py`).

## 3. Export a static build for GitHub Pages

Streamlit's static exporter generates the HTML, JavaScript, and asset files that GitHub Pages can host. Export into the `docs/` folder so GitHub Pages can serve directly from the default `main` branch configuration.

```bash
# Remove any old export to keep the folder clean
rm -rf docs/static_site

# Produce a fresh static build
streamlit static export technical_analysis_app.py --output docs/static_site
```

The export command creates a `docs/static_site` directory containing an `index.html` entry point and all static assets required to run the Streamlit app client-side. Keep this folder under version control.

## 4. (Optional) Smoke test the static build locally

You can quickly preview the exported site using Python's built-in HTTP server:

```bash
cd docs/static_site
python -m http.server 8501
```

Visit `http://localhost:8501` in a browser to confirm the static bundle loads correctly. Stop the server when finished and return to the repository root (`cd ../..`).

## 5. Commit and push the static assets

```bash
git add docs/static_site
git commit -m "Add static Streamlit export for GitHub Pages"
git push origin main
```

If you are using pull requests, push the branch and open a PR instead of committing directly to `main`.

## 6. Configure GitHub Pages

1. Navigate to **Settings → Pages** in your GitHub repository.
2. Under **Build and deployment**, set **Source** to `Deploy from a branch`.
3. Choose the `main` branch and the `/docs` folder.
4. Save the configuration.

GitHub Pages will publish the site at `https://<your-account>.github.io/FinanceAnalysys/` (exact URL depends on your account/organization and custom domain settings). The first deployment may take a couple of minutes.

## 7. Automate future exports (optional)

To keep the static bundle fresh, you can automate the export step by adding a GitHub Actions workflow that runs `streamlit static export` on pushes to `main` or on demand. Commit the generated assets or configure the workflow to push the static files to the `gh-pages` branch.

This repository already ships with `.github/workflows/deploy-pages.yml`, which:

- Restores cached uv artifacts and the project virtual environment to avoid re-downloading wheels.
- Uses `uv add --requirements requirements.txt` instead of `uv pip install` to resolve dependencies in parallel (faster in CI).
- Runs `uv run streamlit static export technical_analysis_app.py --output docs/static_site` to rebuild the static bundle.
- Publishes the exported assets to GitHub Pages using the official `actions/deploy-pages` integration.

Fork owners can reuse this workflow verbatim or adjust the triggers to suit their release cadence.

---

With these steps, your Streamlit-powered technical analysis UI is available through GitHub Pages, enabling stakeholders to explore single-symbol studies, multi-symbol comparisons, and backtesting reports directly from the browser without running Python locally.
