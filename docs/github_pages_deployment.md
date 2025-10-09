# Deploying the Technical Analysis UI to GitHub Pages

This guide explains how to publish the Streamlit-based technical analysis UI to GitHub Pages as a static site. The workflow relies on Streamlit's `export` command (available from Streamlit 1.38.0) to generate HTML, JavaScript, and assets that Pages can host, and shows how to manage the environment with the high-performance [uv](https://docs.astral.sh/uv/latest/) package manager.

## Prerequisites

- Python 3.10 or later (matching the version defined in `pyproject.toml`).
- A fork or clone of the `FinanceAnalysys` repository with push access.
- GitHub Pages enabled for the repository (requires repository admin rights).
- A Streamlit version **≥ 1.38.0**. Earlier releases do not provide the `streamlit export` CLI.
- Node.js is **not** required; the build uses Streamlit's static exporter.

## 1. Set up the development environment

Clone the repository if you have not already:

```bash
git clone https://github.com/<your-account>/FinanceAnalysys.git
cd FinanceAnalysys
```

### Option A — Standard `venv`

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt "streamlit>=1.38.0"
```

### Option B — Using `uv`

`uv` offers very fast dependency resolution and repeatable installs. The steps below create an isolated environment, install dependencies, and run the Streamlit app directly from the `uv` runner.

```bash
# Install uv if you have not already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a dedicated virtual environment (stored in .venv/ by default)
uv venv

# Activate it (optional because uv run/uv pip will auto-activate)
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the project dependencies and ensure Streamlit has export support
uv pip install -r requirements.txt "streamlit>=1.38.0"
```

To launch the application inside the uv-managed environment:

```bash
uv run streamlit run technical_analysis_app.py
```

The `uv run` command automatically boots the environment created above, so you do not need to manually activate it after the first setup.

## 2. Verify the application locally

Whether you used `venv` or `uv`, confirm the UI behaves as expected before exporting:

```bash
streamlit run technical_analysis_app.py
# or, with uv
uv run streamlit run technical_analysis_app.py
```

Open the local URL in your browser, exercise the single-symbol, multi-symbol, and backtesting flows, then stop the server (Ctrl+C) when finished.

## 3. Export a static build for GitHub Pages

Streamlit's exporter generates a fully static bundle. Export into `docs/static_site/` so GitHub Pages can serve it without extra configuration.

```bash
rm -rf docs/static_site
streamlit export technical_analysis_app.py --output docs/static_site
# or, with uv
uv run streamlit export technical_analysis_app.py --output docs/static_site
```

The command produces a folder containing an `index.html` entry point and all required assets. Keep the folder checked in if you deploy manually. The root `.gitignore` already excludes `docs/static_site/` so you can regenerate locally without polluting commits.

## 4. (Optional) Smoke test the static build locally

```bash
cd docs/static_site
python -m http.server 8501
```

Visit `http://localhost:8501` in a browser to confirm the static bundle loads correctly. Stop the server when finished and return to the repository root (`cd ../..`).

## 5. Automate deployments with GitHub Actions

The repository ships with `.github/workflows/deploy_pages.yml`, which builds and deploys the static assets to GitHub Pages automatically. The workflow:

1. Checks out the repository.
2. Installs `uv` and a Streamlit release that supports `streamlit export`.
3. Generates the static bundle under `docs/static_site/` using `uv run streamlit export ...`.
4. Uploads the folder as the Pages artifact and publishes it to the configured Pages environment.

To trigger a deployment, push changes to `main` that touch the app or workflow files, or run the workflow manually via **Actions → Deploy Streamlit UI to GitHub Pages → Run workflow**.

## 6. Configure GitHub Pages

1. Navigate to **Settings → Pages** in your GitHub repository.
2. Under **Build and deployment**, select **GitHub Actions** as the source.
3. (Optional) Set a custom domain or enforce HTTPS as needed.

GitHub will publish the site at `https://<your-account>.github.io/FinanceAnalysys/` (exact URL depends on your account/organization and custom domain settings). The first deployment may take a couple of minutes after the workflow finishes.

---

With these steps, your Streamlit-powered technical analysis UI is available through GitHub Pages, enabling stakeholders to explore single-symbol studies, multi-symbol comparisons, and backtesting reports directly from the browser without running Python locally.
