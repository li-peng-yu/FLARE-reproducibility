#!/usr/bin/env bash
source "$(dirname -- "$0")/common.sh"
SUMMARY="${1:-reports/x5_exact_control_rollout_seed78_20260903.json}"
FIGURE_ROOT="${FIGURE_ROOT:-outputs/figures}"
exec "${PYTHON_BIN}" scripts/render_x5_exact_control_rollout_paper.py \
  --summary "${SUMMARY}" \
  --distribution-output "${FIGURE_ROOT}/appendix_distribution_diagnostics.pdf" \
  --compute-output "${FIGURE_ROOT}/appendix_compute_tradeoffs.pdf"
