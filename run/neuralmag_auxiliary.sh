#!/usr/bin/env bash
source "$(dirname -- "$0")/common.sh"
export ROLLOUT_TARGET_ROOT=""
export SEGMENTS="1 2"
export SCORE_OUTPUT="outputs/auxiliary_distribution/neuralmag_x5"
export EXPECTED_CONDITIONS=66
exec bash run/neuralmag.sh "${@}"
