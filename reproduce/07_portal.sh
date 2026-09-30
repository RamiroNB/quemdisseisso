#!/usr/bin/env bash
# Data of the "Quem disse isso?" portal (app/portal/data/), from the runs and blind rounds above. CPU.
# Nothing is generated or judged here. Then serve the static site locally.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python scripts/export_demo_success.py
uv run python scripts/export_portal_data.py
cd app/portal && python3 serve.py 8803
