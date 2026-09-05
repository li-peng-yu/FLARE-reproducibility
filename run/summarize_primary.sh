#!/usr/bin/env bash
source "$(dirname -- "$0")/common.sh"
REV=outputs/skx_bt_1000base_x5/paper_revision_20260830
GEO=${REV}/exact_control_seed_geometry
ROT=outputs/skx_bt_1000base_x5/paper_revision_20260902/flow_necessity_rotation_mse
QUALITY=${REV}/exact_control_rollout_distribution
EXACT=${REV}/exact_control_rollout_exact_anchor
mkdir -p reports "${EXACT}/fair_energy"
"${PYTHON_BIN}" scripts/evaluate_x5_exact_anchor_forecasts.py \
  --truth-root "${GEO}/distribution/core_s78" \
  --method "flare=${GEO}/exact_anchor/core_s78" \
  --method "cartesian_cfm=${GEO}/exact_anchor/cart_s78" \
  --method "direct_unet=${ROT}/distribution/rotmse_s78" \
  --method "poseidon_t=${QUALITY}/poseidon_t" \
  --method "cno_fm=${QUALITY}/cno_fm" \
  --method "dpot_ti=${QUALITY}/dpot_ti" \
  --method "mpp_avit_ti=${QUALITY}/mpp_avit_ti" \
  --method "pdearena_unet=${QUALITY}/pdearena_unet" \
  --method "le_pde=${QUALITY}/le_pde" \
  --method "neuralmag_x5=${EXACT}/neuralmag_x5" \
  --reference-label flare --output-dir "${EXACT}/fair_energy" \
  --ensemble-size 5 --expected-conditions 33 --bootstrap 50000 --seed 208295700 --workers 8
"${PYTHON_BIN}" scripts/summarize_x5_fixed_2plus3_timing.py \
  --input-root "${REV}/fixed_2plus3_5ns_timing" \
  --output reports/x5_fixed_2plus3_timing_20260830.json
exec "${PYTHON_BIN}" scripts/summarize_x5_exact_control_rollout.py \
  --quality-root "${QUALITY}" \
  --quality-override "flare=${GEO}/distribution/core_s78" \
  --quality-override "cartesian_cfm=${GEO}/distribution/cart_s78" \
  --quality-override "direct_unet=${ROT}/distribution/rotmse_s78" \
  --fair-energy-root "${EXACT}/fair_energy" \
  --fair-energy-override "flare=core_s78=${GEO}/fair_energy/run_summary.json" \
  --fair-energy-override "cartesian_cfm=cart_s78=${GEO}/fair_energy/run_summary.json" \
  --fair-energy-override "direct_unet=rotmse_s78=${ROT}/fair_energy/run_summary.json" \
  --fixed-two-segment-timing-summary reports/x5_fixed_2plus3_timing_20260830.json \
  --output reports/x5_exact_control_rollout_seed78_20260903.json
