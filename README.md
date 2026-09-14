# dingo_ws

Clean ROS 2 Jazzy workspace for the Clearpath Dingo-D DD100.

## Official Dingo-D DD100 reference

The robot profile is kept in `src/dingo_bringup/config/dingo_dd100.yaml`:

- dimensions (L x W x H): **551 x 517 x 110 mm**
- ground clearance: **14 mm**
- mass: **12 kg**
- maximum payload: **20 kg**
- maximum speed: **1.3 m/s**
- drive: differential

These values come from Clearpath's Dingo-D user manual. Clearpath's current
ROS documentation lists DD100 as supported on ROS 2 Jazzy.

This workspace is intentionally separate from the old `robot_ws`. The source
tree contains only the fresh `dingo_bringup` package; nothing from the old
robot stack is copied here. Generated `build/`, `install/`, and `log/`
directories are created locally by colcon and are ignored by Git.

## First build

```bash
cd /home/dimi/dingo_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## First four stages

The official Clearpath platform services must be running first. The clean
sensor stack also starts the mounted-sensor TF and then the Hokuyo and D455:

These launch commands are for isolated diagnostics only. Do not run them while
the managed services below are active; the mounted sensor stack and Dashboard
have single-instance locks and will reject a second copy.

After the Dingo hardware is connected, the mounted sensor stack is started
with one command from this workspace:

Use the managed command instead:

```bash
dingo start
```

This starts the clean Hokuyo and RealSense launches. The official Clearpath
platform services and the new Dashboard remain separate, so they are not
started a second time. The Dashboard and sensor stack reject a second manual
copy, and SLAM/Nav2 are mutually exclusive.

For the normal one-command startup, open a new terminal and type:

```bash
dingo
```

The mounted Hokuyo and RealSense stack is supervised by the user service
`dingo-sensors.service`, so `/scan`, the sensor TF, and the camera recover if
the launch process fails. Enable it once with:

```bash
systemctl --user enable --now dingo-sensors.service
```

This command starts the Clearpath base if needed and ensures the Dashboard and
sensor service are running. To inspect the single running stack, use
`dingo status`; to stop everything, use `dingo stop` from another terminal.
The localization process uses the official Clearpath Nav2 launch
with `map:=dingo_map` and `/scan`; AMCL continuously publishes `map -> odom`,
while the platform publishes `odom -> base_link`.

Open `http://localhost:8090` for the new dashboard. Keep the STOP button and
the physical emergency stop available whenever teleop is enabled.

## End-to-end preflight

The clean workspace includes a read-only preflight checker. It never starts or
stops ROS nodes and never publishes a motion command:

```bash
source /opt/ros/jazzy/setup.bash
source /home/dimi/dingo_ws/install/setup.bash
ros2 run dingo_bringup preflight.sh --quick
ros2 run dingo_bringup preflight.sh --mapping
ros2 run dingo_bringup preflight.sh --navigation
```

Run the baseline check after `dingo` starts. Use `--mapping` only while SLAM is
running, and `--navigation` only after switching to a saved map and starting
Nav2. SLAM and navigation are separate modes.

## Dashboard v2

The Dashboard has separate views for overview, mapping, driving, sensors, and
camera/LiDAR perception. The Mapping view can start and stop SLAM, save a map
with a name, and place room labels by clicking on the map. Saved maps go to
`/home/dimi/dingo_ws/maps`; room labels are kept in
`~/.config/dingo_dashboard/rooms.json`.

At boot, the Dashboard automatically starts localization/Nav2 with
`dingo_map` once odometry, LiDAR and mounted-sensor TF are available. It does
not send a navigation goal. It first calls the official AMCL
`reinitialize_global_localization` service and waits for several stable,
low-covariance AMCL updates. A stationary LiDAR-to-map match is only used as a
fallback hypothesis and is accepted only after AMCL confirms it. Boot never
commands a rotation; an explicit `Αυτόματη εύρεση` request may allow the slow
in-place search after confirmation. To use SLAM instead, stop Nav2 before
starting a mapping session. For autonomous navigation, choose a saved map in the Mapping view and press
`Ενεργοποίηση Nav2`. Wait until the status is ready. A pose is transferred
automatically only when switching directly from the currently running SLAM
session; an old transform from a previous run is never reused. Press
`Αυτόματη εύρεση` first performs the official AMCL global reset and waits for
the AMCL pose to stabilize. If needed, a conservative LiDAR-to-map scan match
is handed to AMCL as a fallback and is not trusted by itself. If global search
cannot disambiguate the rooms, press
`Ορισμός θέσης`, then press on the Dingo's position and drag in the direction
it is facing. After localization is confirmed, click an empty, known cell on
the map to send a `NavigateToPose` goal; the planner path is drawn on the map
in green as soon as Nav2 returns it. Use `Ακύρωση στόχου` for the current
goal or `Κλείσιμο Nav2` to stop localization and navigation. SLAM and
autonomous navigation are separate modes and must not run at the same time.
When a goal, patrol, Spin, or Drive-on-Heading action is explicitly started
from the Dashboard or voice assistant, the Dashboard temporarily releases
only the BT-quality lock through the fail-closed gate topic, so a sleeping
Bluetooth PS5 cannot block Nav2. The physical emergency-stop and safety-stop
locks remain at priorities 255 and 254. On completion, cancellation, shutdown,
or service restart the gate returns to locked automatically; the Mapping view
shows the current mode.
The gate is persisted in the Clearpath generator input at
`/etc/clearpath/robot.yaml` under
`platform.extras.ros_parameters.twist_mux.locks.bt_quality` (topic
`joy_teleop/bt_quality_stop_gate`, timeout `0.5`). If Clearpath parameters are
regenerated, keep that block and run
`/opt/ros/jazzy/lib/clearpath_generator_robot/generate_param -s /etc/clearpath`
before restarting the platform service. The reference overlay is kept in
`systemd/twist_mux-autonomous.yaml`.
The overview and Sensors views also show live Dingo battery input and the raw
12 V current-sense value when `/platform/mcu/status/power` is published. The
external Mini PC is powered separately and is therefore outside this total.
The power message does not expose the MCU chip's direct consumption or a
motor-only wattage, so those values are not inferred from the rail reading.

While AMCL has no trusted pose, the Dashboard deliberately hides the
map-projected LiDAR points and the robot marker. This prevents an arbitrary
`map -> odom` transform after a failed global search from looking like a real
wall-registration error.

## Voice assistant

The optional voice service is prepared for the Seeed Studio ReSpeaker XMOS
XVF3800. It looks only for a device whose name contains `ReSpeaker`; while the
array is disconnected it stays in `waiting_for_microphone` and never falls
back to the Mini PC's built-in microphone.

The pipeline is deliberately constrained: ReSpeaker audio is published as
`foxglove_msgs/msg/RawAudio`, Whisper medium performs Greek speech-to-text on
the AMD Ryzen AI NPU through VitisAI, and
the configured LLM uses allow-listed function/tool calls for robot and system
actions when `Native tool calling` is enabled from the Dashboard. It is
disabled by default, so the faster validated JSON intent format is used.
Replies are spoken with Gemini
TTS (`gemini-3.1-flash-tts-preview`) using a natural female Greek voice; the 2.5 Flash
TTS model is an optional cloud fallback. The local Piper backup voice has been
removed. The Dashboard is the only
node allowed to execute a robot action. Navigation requires a second spoken
confirmation (`ναι`); `σταμάτα` is immediate. The LLM cannot run shell
commands or publish velocity.

The voice node uses channel 0 from the current direct-ALSA XVF3800 endpoint
for VAD and Whisper and does not average the six USB channels into one signal.
The official English Alexa detector is retained as a fast path, while a
VAD-complete accelerated Whisper check verifies the Greek pronunciations
«Αλέξα» and «Αλέχα» before any transcript can reach the assistant. Its VAD
also requires several consecutive active audio blocks, which prevents short
room noise from starting empty command jobs.

The wake phrase is `Hey Dingo`. Its local detector is trained from the
ReSpeaker recordings in `training/wake_word_dingo`; the model is not enabled
until real positive and hard-negative examples pass evaluation.

The voice service starts with `dingo`/`dingo start` and can be inspected with:

```bash
systemctl --user status dingo-voice.service
ros2 topic echo /dd100_10000002/voice/status
ros2 topic echo /dd100_10000002/voice/transcript
ros2 topic echo /dd100_10000002/voice/reply
```

The Dashboard also has a `Βοηθός` tab for typed Greek questions and commands.
Typed requests use the same Gemini, allowlist and Dashboard executor as spoken
requests; navigation still requires a second `ναι` confirmation. The HTTP
entry point is `POST /api/assistant` with `{"text":"..."}`. The Dashboard
does not execute shell commands from assistant text.

The configured model is `gemini-2.5-flash`. Create a Gemini API key in Google
AI Studio and store only the key (not `GEMINI_API_KEY=...`) in
`/home/dimi/.config/dingo_voice/gemini_api_key` with mode `600`, then restart
`dingo-voice.service`. Do not paste the key into chat or commit it. The old
Qwen model is not deleted, but Ollama is no longer started by `dingo`; this
keeps one active LLM path. The voice service uses `whisper.cpp`
`large-v3-turbo` on the AMD Radeon GPU through Vulkan. If the Vulkan
executable or GPU is unavailable, it falls back automatically to CPU
`large-v3-turbo`. The AMD NPU Whisper backend remains available as an
explicit alternative. Do not run a second voice process manually.

The `API` view and these read-only HTTP endpoints expose the live Clearpath
ROS 2 graph for the DD100: `/api/clearpath`, `/api/clearpath/topics`,
`/api/clearpath/services`, `/api/clearpath/actions`, and
`/api/clearpath/status`. They include the official platform/MCU/sensor
interfaces, active ROS types, QoS labels, publishers/subscribers, Nav2 actions,
nodes and live telemetry. Motion and MCU command interfaces are listed for
visibility but remain guarded by the existing safety controls.

The perception view reads the RealSense color image from
`/camera/camera/color/image_raw` and draws nearby LiDAR points from `/scan`.
The «Ενεργοποίηση κάμερας» button starts the isolated RealSense launch. The
SLAM wrapper uses `/scan` and publishes the Clearpath namespaced map at
`/dd100_10000002/map`.

## Mounted sensors

The confirmed hardware profile is kept separately in
`src/dingo_bringup/config/sensor_profile.yaml`:

- Hokuyo **UTM-30LX-EW**, product code **UUTM013**: enclosure **62 × 62 ×
  87.5 mm**, 270° scan, 0.25° angular resolution and 30 m guaranteed range.
- Intel RealSense **D455**: enclosure **124 × 29 × 26 mm**, depth field of
  view **86° × 57°** and manufacturer working range **0.6–6 m**.

The clean stack now publishes the initial mounted-sensor TF in the same
Clearpath namespace as the platform:

- `base_link -> laser` on `/dd100_10000002/tf_static`
- `base_link -> camera_link -> camera_color_optical_frame` on
  `/dd100_10000002/tf_static`

The initial `x`, `y`, `z`, yaw, pitch and roll values are kept in
`src/dingo_bringup/config/sensor_profile.yaml` and are marked
`provisional_from_photos`. They should be refined if a real mapping run shows
registration error; the sensor enclosure dimensions alone are not TF origins.

The SLAM launch now uses the live `/scan` topic and the namespaced platform TF.
The Dashboard map saver waits for DDS discovery before writing `.pgm` and
`.yaml` files, so named map saving is ready for a real mapping run.

The Driving view has a persistent maximum-speed slider from **0.02 to 0.25
m/s**. The limit is enforced in the ROS dashboard node as well as in the web
interface, and is stored in `~/.config/dingo_dashboard/settings.json`.

For remote access over the already-configured Tailscale VPN, start it bound
only to the Tailscale interface:

```bash
ros2 launch dingo_bringup dashboard.launch.py host:=100.88.95.48
```

Then open the HTTPS Tailscale URL shown by `tailscale serve status`, followed
by `/dingo/`, on a phone signed in to the same Tailscale network. Do not use
router port forwarding for the dashboard while it has motion controls.
