#!/usr/bin/env bash
# Start the detection API for the Chrome extension.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m phishnet.serve.api
