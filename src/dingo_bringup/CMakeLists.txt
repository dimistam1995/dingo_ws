cmake_minimum_required(VERSION 3.8)
project(dingo_bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch web DESTINATION share/${PROJECT_NAME})
# The fresh identity models live at the workspace level so they can also be
# inspected/downloaded without touching the ROS package layout.
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../models/
  DESTINATION share/${PROJECT_NAME}/models
)
install(PROGRAMS
  scripts/dingo_dashboard.py
  scripts/dingo_voice_assistant.py
  scripts/dingo_npu_whisper_client.py
  scripts/dingo_npu_whisper_worker.py
  scripts/dingo_mic_array.py
  scripts/dingo_face_recognition.py
  scripts/dingo_speaker_identity.py
  scripts/preflight.sh
  scripts/rviz_dashboard_bridge.sh
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
