#!/usr/bin/env bash
# Restore Clearpath's normal fail-closed Bluetooth-quality lock.
# This is deliberately a fixed parameter update; it never touches the
# physical emergency-stop or safety-stop priorities.
set -e

namespace="${1:-dd100_10000002}"
if [[ ! "$namespace" =~ ^[A-Za-z0-9_/-]+$ ]]; then
    echo "invalid robot namespace" >&2
    exit 2
fi

source /opt/ros/jazzy/setup.bash
source /home/dimi/dingo_ws/install/setup.bash
exec /opt/ros/jazzy/bin/ros2 param set "/${namespace#/}/twist_mux" \
    locks.bt_quality.priority 253
