#!/usr/bin/env bash
source "$(dirname -- "$0")/common.sh"
METHOD="${1:?Usage: benchmark_batch_scaling.sh METHOD [NEURALMAG_BATCH_INDEX]}"
case "${METHOD}" in
  flare) INDEX=0 ;;
  poseidon_t) INDEX=1 ;;
  cno_fm) INDEX=2 ;;
  dpot_ti) INDEX=3 ;;
  mpp_avit_ti) INDEX=4 ;;
  pdearena_unet) INDEX=5 ;;
  le_pde) INDEX=6 ;;
  neuralmag_x5)
    exec bash run/experiments/run_x5_fixed_2plus3_neuralmag_sweep.sh "${2:?Supply batch index 0..7}" ;;
  *) echo "Unknown method: ${METHOD}" >&2; exit 2 ;;
esac
exec bash run/experiments/run_x5_fixed_2plus3_batch_sweep.sh "${INDEX}"
