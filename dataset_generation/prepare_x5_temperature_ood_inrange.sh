#!/usr/bin/env bash
source "$(dirname -- "$0")/../run/common.sh"
exec "${PYTHON_BIN}" run/temperature.py prepare --temperatures 400 30 90 150 225 "$@"
