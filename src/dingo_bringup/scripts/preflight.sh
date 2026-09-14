#!/usr/bin/env bash
# Read-only pre-flight check for the clean Clearpath Dingo workspace.
#
# Modes:
#   preflight.sh              sensor/base baseline
#   preflight.sh --mapping    baseline + active SLAM/map checks
#   preflight.sh --navigation baseline + Nav2/localization checks
#   preflight.sh --quick      skip rate probes
#
# The script only queries ROS, systemd and the network. It never publishes a
# motion command, starts/stops a node, changes parameters, or resets the robot.
set -uo pipefail

readonly SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
readonly PACKAGE_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")/.." && pwd)"
readonly WORKSPACE="$(cd -- "$PACKAGE_DIR/../.." && pwd)"
readonly NAMESPACE="${DINGO_NAMESPACE:-dd100_10000002}"
readonly LIDAR_IP="${DINGO_LIDAR_IP:-192.168.0.10}"
readonly LIDAR_PORT="${DINGO_LIDAR_PORT:-10940}"
readonly BASE="/${NAMESPACE}"
readonly ODOM_TOPIC="${BASE}/platform/odom"
readonly BATTERY_TOPIC="${BASE}/platform/bms/state"
readonly ESTOP_TOPIC="${BASE}/platform/emergency_stop"
readonly SAFETY_TOPIC="${BASE}/platform/safety_stop"
readonly MCU_STOP_TOPIC="${BASE}/platform/mcu/status/stop"
readonly MOTORS_TOPIC="${BASE}/platform/motors/status"
readonly DIAGNOSTICS_TOPIC="${BASE}/diagnostics_agg"
readonly MAP_TOPIC="${BASE}/map"
readonly ACTION="${BASE}/navigate_to_pose"

MODE="sensors"
QUICK=false
PASS=0
WARN=0
FAIL=0

ok() {
  echo "  ✔ $1"
  PASS=$((PASS + 1))
}

warn() {
  echo "  ⚠ $1"
  WARN=$((WARN + 1))
}

bad() {
  echo "  ✖ $1"
  FAIL=$((FAIL + 1))
}

usage() {
  sed -n '1,18p' "$0"
  echo
  echo "Examples:"
  echo "  $0"
  echo "  $0 --quick"
  echo "  $0 --mapping"
  echo "  $0 --navigation"
}

while (($# > 0)); do
  case "$1" in
    --mapping) MODE="mapping" ;;
    --navigation) MODE="navigation" ;;
    --quick) QUICK=true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Άγνωστη επιλογή: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if ! command -v ros2 >/dev/null 2>&1; then
  bad "ros2 CLI δεν είναι διαθέσιμο"
  echo "Σύνοψη: $PASS ✔  $WARN ⚠  $FAIL ✖"
  exit 1
fi

topic_publishers() {
  local topic="$1"
  timeout 8 ros2 topic info "$topic" 2>/dev/null \
    | awk -F: '/Publisher count:/ {gsub(/[[:space:]]/, "", $2); print $2; exit}'
}

has_publisher() {
  local topic="$1" count
  count="$(topic_publishers "$topic")"
  [[ "${count:-0}" =~ ^[1-9][0-9]*$ ]]
}

sample_topic() {
  local topic="$1"
  timeout 8 ros2 topic echo "$topic" --once \
    --qos-reliability best_effort >/dev/null 2>&1
}

rate_topic() {
  local topic="$1" minimum="$2" rate
  rate="$(timeout 10 ros2 topic hz "$topic" --window 5 2>/dev/null \
    | awk '/average rate:/ {print $3; exit}')"
  [[ -n "${rate:-}" ]] || return 1
  awk -v value="$rate" -v min="$minimum" 'BEGIN {exit !(value >= min)}'
}

check_topic() {
  local topic="$1" label="$2"
  if has_publisher "$topic"; then
    ok "$label: publisher"
    if sample_topic "$topic"; then
      ok "$label: δεδομένα"
    else
      bad "$label: publisher χωρίς δείγμα"
    fi
  else
    bad "$label: χωρίς publisher"
  fi
}

check_rate() {
  local topic="$1" minimum="$2" label="$3" rate
  rate="$(timeout 10 ros2 topic hz "$topic" --window 5 2>/dev/null \
    | awk '/average rate:/ {print $3; exit}')"
  if [[ -z "${rate:-}" ]]; then
    bad "$label: δεν μετρήθηκε συχνότητα"
  elif awk -v value="$rate" -v min="$minimum" 'BEGIN {exit !(value >= min)}'; then
    ok "$label: ${rate} Hz"
  else
    bad "$label: μόνο ${rate} Hz"
  fi
}

check_service() {
  local service="$1" label="$2" user_service="${3:-false}"
  if [[ "$user_service" == true ]]; then
    systemctl --user is-active --quiet "$service" \
      && ok "$label ενεργό" \
      || bad "$label ανενεργό"
  else
    systemctl is-active --quiet "$service" \
      && ok "$label ενεργό" \
      || bad "$label ανενεργό"
  fi
}

check_tf() {
  local parent="$1" child="$2" label="$3" output
  output="$(timeout 8 ros2 run tf2_ros tf2_echo "$parent" "$child" \
    --ros-args -r /tf:="${BASE}/tf" -r /tf_static:="${BASE}/tf_static" \
    2>&1 || true)"
  if grep -qE 'Translation|At time' <<<"$output"; then
    ok "$label"
  else
    bad "$label λείπει"
  fi
}

echo "══ Dingo preflight (${MODE}) ══"
echo "Workspace: $WORKSPACE"
echo "Namespace: $NAMESPACE"

echo "── Εγκατάσταση"
if [[ -f "$WORKSPACE/install/setup.bash" ]] && ros2 pkg prefix dingo_bringup >/dev/null 2>&1; then
  ok "dingo_bringup εγκατεστημένο"
else
  bad "dingo_bringup δεν είναι εγκατεστημένο — κάνε colcon build"
fi
if command -v dingo >/dev/null 2>&1; then
  ok "dingo helper διαθέσιμο"
else
  bad "dingo helper δεν βρέθηκε στο PATH"
fi

echo "── Clearpath services"
check_service clearpath-robot.service "clearpath-robot.service"
check_service clearpath-platform.service "clearpath-platform.service"
check_service clearpath-vcan.service "clearpath-vcan.service"
check_service clearpath-shutdown.service "clearpath-shutdown.service"
check_service dingo-dashboard.service "dingo-dashboard.service" true

echo "── Δίκτυο και αισθητήρες"
if timeout 3 bash -c "cat </dev/null > /dev/tcp/${LIDAR_IP}/${LIDAR_PORT}" 2>/dev/null; then
  ok "Hokuyo ${LIDAR_IP}:${LIDAR_PORT} reachable"
elif has_publisher "/scan"; then
  ok "Hokuyo active driver owns ${LIDAR_IP}:${LIDAR_PORT}; /scan is healthy"
else
  bad "Hokuyo ${LIDAR_IP}:${LIDAR_PORT} unreachable"
fi
check_topic "$ODOM_TOPIC" "Odometry $ODOM_TOPIC"
check_topic "$BATTERY_TOPIC" "Battery $BATTERY_TOPIC"
check_topic "$ESTOP_TOPIC" "Emergency stop $ESTOP_TOPIC"
if has_publisher "$SAFETY_TOPIC"; then
  if sample_topic "$SAFETY_TOPIC"; then ok "Safety stop δεδομένα"; else warn "Safety stop publisher χωρίς δείγμα"; fi
else
  warn "Safety stop topic χωρίς publisher"
fi
check_topic "$MCU_STOP_TOPIC" "MCU stop status"
check_topic "$MOTORS_TOPIC" "Motor status"
check_topic "/camera/camera/color/image_raw" "RealSense color"
check_topic "/camera/camera/depth/image_rect_raw" "RealSense depth"
check_topic "/scan" "Hokuyo /scan"

echo "── Ασφάλεια και diagnostics"
estop="$(timeout 8 ros2 topic echo "$ESTOP_TOPIC" --once \
  --qos-reliability best_effort 2>/dev/null \
  | awk -F: '/^data:/ {gsub(/[[:space:]]/, "", $2); print $2; exit}')"
case "$estop" in
  false) ok "E-stop απελευθερωμένο" ;;
  true) bad "E-stop ενεργό — δεν επιτρέπεται κίνηση" ;;
  *) bad "E-stop status δεν διαβάστηκε" ;;
esac
diagnostics_sample="$(timeout 8 ros2 topic echo "$DIAGNOSTICS_TOPIC" --once \
  --qos-reliability best_effort 2>/dev/null || true)"
if [[ -n "$diagnostics_sample" ]]; then
  ok "Clearpath diagnostics απαντούν"
  if grep -qiE 'High execution jitter|Frequency too low|CAN Receive Timeout|message: Error|level: 2' <<<"$diagnostics_sample"; then
    bad "Clearpath diagnostics αναφέρουν ενεργό σφάλμα ή υποβάθμιση"
  else
    ok "Clearpath diagnostics χωρίς ενεργό error"
  fi
else
  bad "Clearpath diagnostics δεν απαντούν"
fi

echo "── ROS graph και TF"
nodes="$(timeout 10 ros2 node list 2>/dev/null || true)"
if [[ -n "$nodes" ]]; then
  ok "ROS graph απαντά ($(wc -l <<<"$nodes") nodes)"
else
  bad "ROS graph δεν απαντά"
fi
duplicates="$(printf '%s\n' "$nodes" | sort | uniq -d)"
if [[ -z "$duplicates" ]]; then
  ok "κανένα διπλό node"
else
  bad "διπλά nodes: $(tr '\n' ' ' <<<"$duplicates")"
fi
check_tf odom base_link "TF odom → base_link"
check_tf base_link laser "TF base_link → laser"
check_tf base_link camera_link "TF base_link → camera_link"

if [[ "$QUICK" == false ]]; then
  echo "── Συχνότητες"
  check_rate "/scan" 5 "Hokuyo /scan"
  check_rate "$ODOM_TOPIC" 5 "Odometry"
  check_rate "/camera/camera/color/image_raw" 5 "RealSense color"
fi

if [[ "$MODE" == mapping ]]; then
  echo "── SLAM / mapping"
  if has_publisher "$MAP_TOPIC"; then
    ok "SLAM map publisher $MAP_TOPIC"
    sample_topic "$MAP_TOPIC" && ok "SLAM map δεδομένα" || bad "SLAM map χωρίς δείγμα"
  else
    bad "SLAM map $MAP_TOPIC χωρίς publisher"
  fi
  if printf '%s\n' "$nodes" | grep -qE '/(slam_toolbox|sync_slam_toolbox)$'; then
    ok "slam_toolbox node ενεργό"
  else
    bad "slam_toolbox node δεν βρέθηκε"
  fi
  check_tf map odom "TF map → odom (SLAM)"
elif [[ "$MODE" == navigation ]]; then
  echo "── Nav2 / localization"
  if timeout 8 ros2 action info "$ACTION" 2>/dev/null | grep -q 'Action clients\|Action servers'; then
    ok "NavigateToPose action διαθέσιμο"
  else
    bad "NavigateToPose action δεν είναι διαθέσιμο"
  fi
  for node in map_server amcl bt_navigator controller_server planner_server behavior_server; do
    # Lifecycle services can need a few seconds for DDS discovery after Nav2
    # starts, especially while map_server is loading a map.
    state="$(timeout 12 ros2 lifecycle get "${BASE}/${node}" 2>/dev/null | tail -1)"
    if grep -q 'active' <<<"$state"; then
      ok "${node}: active"
    else
      bad "${node}: ${state:-δεν αποκρίνεται}"
    fi
  done
  if has_publisher "$MAP_TOPIC"; then
    ok "Nav2 map publisher $MAP_TOPIC"
  else
    bad "Nav2 map $MAP_TOPIC χωρίς publisher"
  fi
  check_tf map odom "TF map → odom (localization)"
fi

echo
echo "Σύνοψη: $PASS ✔  $WARN ⚠  $FAIL ✖"
if ((FAIL == 0)); then
  echo "Το ${MODE} preflight είναι επιτυχές."
  exit 0
fi
echo "Το ${MODE} preflight έχει εκκρεμότητες. Δεν δόθηκε εντολή κίνησης."
exit 1
