#!/usr/bin/env bash
source "$(dirname -- "$0")/common.sh"
AUDIT_ROOT="${AUDIT_ROOT:-outputs/skx_bt_165base_x30/evaluation_20260828/standard_prior50k_on_x30_test}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-outputs/paper_artifacts/generative_necessity}"
FIGURE_ROOT="${FIGURE_ROOT:-outputs/figures/generative_necessity}"
mkdir -p "${ARTIFACT_ROOT}" "${FIGURE_ROOT}"
"${PYTHON_BIN}" scripts/make_generative_necessity_figure.py select \
  --audit-root "${AUDIT_ROOT}" --output-dir "${ARTIFACT_ROOT}" --no-paper-copy
# The supplied bundle includes the original-versus-compact tensor audit.
# Re-run that audit when original full training checkpoints are supplied.
EQUIVALENCE_ARGS=(--skip-table1-equivalence-audit)
if [[ -n "${TABLE1_ORIGINAL_POSEIDON:-}" && -n "${TABLE1_ORIGINAL_FLARE:-}" ]]; then
  EQUIVALENCE_ARGS=(--table1-poseidon-checkpoint "${TABLE1_ORIGINAL_POSEIDON}"
                    --table1-flare-checkpoint "${TABLE1_ORIGINAL_FLARE}")
fi
"${PYTHON_BIN}" scripts/export_generative_necessity_poseidon.py \
  --selection "${ARTIFACT_ROOT}/selection_manifest.json" --output-dir "${ARTIFACT_ROOT}" \
  --config configs/data/x30_evaluation.yaml --audit-root "${AUDIT_ROOT}" \
  --poseidon-checkpoint "${FLARE_CHECKPOINT_ROOT}/baselines/poseidon_t.pt" \
  --flare-checkpoint "${FLARE_CHECKPOINT_ROOT}/flare/core_seed78.pt" \
  --stats configs/stats/dataset_stats_both.json --device "${DEVICE}" --precision fp32 \
  --seed 208310700 --flare-seed 2026083100 "${EQUIVALENCE_ARGS[@]}"
"${PYTHON_BIN}" scripts/make_generative_necessity_figure.py audit \
  --audit-root "${AUDIT_ROOT}" --output-dir "${ARTIFACT_ROOT}" --no-paper-copy
"${PYTHON_BIN}" scripts/make_generative_necessity_figure.py render-main \
  --audit-root "${AUDIT_ROOT}" --output-dir "${ARTIFACT_ROOT}" \
  --figure-output-dir "${FIGURE_ROOT}/main_q_mz" --main-scatter-y mean_mz --no-paper-copy
exec "${PYTHON_BIN}" scripts/make_generative_necessity_figure.py render-appendix \
  --audit-root "${AUDIT_ROOT}" --output-dir "${ARTIFACT_ROOT}" \
  --figure-output-dir "${FIGURE_ROOT}/appendix" --no-paper-copy
