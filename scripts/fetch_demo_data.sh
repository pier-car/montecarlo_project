#!/usr/bin/env bash
# Generate the small synthetic demo dataset shipped with NEPHELE.
#
# The demo intentionally uses a synthetic, license-clean city block rather than
# real geodata so the repository stays self-contained and export-friendly.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_DIR="${ROOT}/data/demo"
mkdir -p "${DEMO_DIR}"

echo "Demo data already vendored in ${DEMO_DIR}:"
ls -1 "${DEMO_DIR}"
echo "Run a scenario with:"
echo "  python scripts/run_scenario.py --config ${DEMO_DIR}/release.yaml"
