#!/usr/bin/env bash
# env-setup.sh — helper to install requirements and the editable package into the included conda Python
set -euo pipefail
PYTHON=./.conda/bin/python

echo "Installing runtime requirements into ${PYTHON}..."
${PYTHON} -m pip install -r requirements.txt

echo "Installing local package in editable mode..."
${PYTHON} -m pip install -e ./geoclip_og

echo "Done. To run examples, use the same interpreter: ${PYTHON} -m examples.basic_inference or runpy.run_path(...)"
