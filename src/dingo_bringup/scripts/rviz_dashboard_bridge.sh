#!/usr/bin/env bash
set -Eeuo pipefail

# RViz2/Jazzy currently crashes against the AMD Radeon 860M GLX stack on the
# host.  Run the real RViz2 process on a software X display and expose that
# display to the Dashboard through the standard noVNC client.
DISPLAY_ID="${RVIZ_DISPLAY:-:99}"
VNC_PORT="${RVIZ_VNC_PORT:-5909}"
WEB_PORT="${RVIZ_WEB_PORT:-8091}"
ROS2_BIN="${ROS2_BIN:-/opt/ros/jazzy/bin/ros2}"
RVIZ_CONFIG="${RVIZ_CONFIG:-$($ROS2_BIN pkg prefix dingo_bringup)/share/dingo_bringup/config/dingo.rviz}"
RVIZ_NAMESPACE="${RVIZ_NAMESPACE:-dd100_10000002}"

xvfb_pid=""
x11vnc_pid=""
websockify_pid=""
rviz_pid=""

cleanup() {
  trap - EXIT INT TERM
  for pid in "$rviz_pid" "$websockify_pid" "$x11vnc_pid" "$xvfb_pid"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  wait "$rviz_pid" "$websockify_pid" "$x11vnc_pid" "$xvfb_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if xdpyinfo -display "$DISPLAY_ID" >/dev/null 2>&1; then
  echo "RViz display $DISPLAY_ID is already in use" >&2
  exit 1
fi

Xvfb "$DISPLAY_ID" -screen 0 1280x900x24 -nolisten tcp +extension GLX +render -noreset \
  >/dev/null 2>&1 &
xvfb_pid=$!
for _ in {1..40}; do
  if xdpyinfo -display "$DISPLAY_ID" >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done
xdpyinfo -display "$DISPLAY_ID" >/dev/null 2>&1

x11vnc -display "$DISPLAY_ID" -rfbport "$VNC_PORT" -localhost -forever -shared -nopw -quiet \
  >/dev/null 2>&1 &
x11vnc_pid=$!

/usr/bin/websockify --web=/usr/share/novnc --heartbeat=30 \
  "$WEB_PORT" "127.0.0.1:$VNC_PORT" >/dev/null 2>&1 &
websockify_pid=$!

export DISPLAY="$DISPLAY_ID"
export QT_QPA_PLATFORM=xcb
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
export GALLIUM_DRIVER=llvmpipe

/opt/ros/jazzy/lib/rviz2/rviz2 \
  -d "$RVIZ_CONFIG" \
  --ros-args \
  -r "/tf:=/$RVIZ_NAMESPACE/tf" \
  -r "/tf_static:=/$RVIZ_NAMESPACE/tf_static" &
rviz_pid=$!
wait "$rviz_pid"
