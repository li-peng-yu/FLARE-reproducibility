#!/usr/bin/env bash
source "$(dirname -- "$0")/common.sh"
METHOD="${1:?Usage: evaluate_distribution.sh METHOD}"
case "${METHOD}" in
  flare) SCRIPT=run_x5_seed_geometry_exact_control_rollout; INDEX=0 ;;
  cartesian_cfm) SCRIPT=run_x5_seed_geometry_exact_control_rollout; INDEX=3 ;;
  direct_unet) SCRIPT=run_x5_rotation_mse_exact_control; INDEX=0 ;;
  poseidon_t) SCRIPT=run_x5_multisegment_rollout_distribution; INDEX=1 ;;
  cno_fm) SCRIPT=run_x5_multisegment_rollout_distribution; INDEX=2 ;;
  dpot_ti) SCRIPT=run_x5_multisegment_rollout_distribution; INDEX=3 ;;
  mpp_avit_ti) SCRIPT=run_x5_multisegment_rollout_distribution; INDEX=4 ;;
  pdearena_unet) SCRIPT=run_x5_multisegment_rollout_distribution; INDEX=5 ;;
  le_pde) SCRIPT=run_x5_multisegment_rollout_distribution; INDEX=6 ;;
  neuralmag_x5) SCRIPT=run_x5_multisegment_rollout_neuralmag; INDEX="${2:?Supply shard 0..10}" ;;
  *) echo "Unknown method: ${METHOD}" >&2; exit 2 ;;
esac
exec bash "run/experiments/${SCRIPT}.sh" "${INDEX}"
