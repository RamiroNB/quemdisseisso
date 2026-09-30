#!/usr/bin/env bash
# Speaker-tagged evidence and derived speaker attributes. CPU.
# derive_speaker_attributes.py queries the Chamber of Deputies open-data API and Wikipedia.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python scripts/build_tagged_evidence.py
uv run python scripts/derive_speaker_attributes.py
