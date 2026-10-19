# Nova Carter Perception

ROS 2 perception pipeline for the **Nova Carter** robot in **NVIDIA Isaac Sim**, combining:

- Left + right RGB cameras
- YOLOv8n object detection
- 3D LiDAR
- ROS 2 TF transforms
- Camera–LiDAR fusion
- Multi-camera 3D association
- Predefined robot motion
- CSV logging and annotated video export

Repository:  
https://github.com/youssefMohameedd/Nova_Carter_Perception

---

## Overview

The project was developed incrementally:

1. Validate ROS 2 communication with Isaac Sim.
2. Receive RGB camera images and camera calibration data.
3. Run YOLOv8n on the left camera.
4. Transform 3D LiDAR points into the camera optical frame using TF.
5. Project LiDAR points into the RGB image using camera intrinsics.
6. Associate LiDAR points with YOLO bounding boxes.
7. Estimate object XYZ position and range.
8. Publish standard `vision_msgs/msg/Detection3DArray` results.
9. Add a second front camera and a second YOLO + LiDAR fusion pipeline.
10. Fuse left/right 3D detections in the common `base_link` frame.
11. Move the robot through a predefined trajectory while recording detections and video.

A useful TF validation result was:

```text
LiDAR -> LEFT camera  : x ≈ +0.075 m
LiDAR -> RIGHT camera : x ≈ -0.075 m
```

This confirms the expected mirrored placement of the two front cameras around the LiDAR.

---

## Main Architecture

```text
Left Camera  -> YOLO -> Left Camera/LiDAR Fusion  --                                                      -> Multi-Camera Fusion
Right Camera -> YOLO -> Right Camera/LiDAR Fusion --/
                         ^
                         |
                      3D LiDAR

Final output:
    /fusion/final/detections_3d
    vision_msgs/msg/Detection3DArray
```

Robot motion:

```text
/chassis/odom -> trajectory_controller -> /cmd_vel
```

---

## Main ROS 2 Nodes

| Node | Purpose |
|---|---|
| `yolo_detector` | YOLO detection from left camera |
| `right_yolo_detector` | YOLO detection from right camera |
| `lidar_camera_projection` | LiDAR-to-camera projection debugging |
| `lidar_yolo_fusion` | Left camera + LiDAR 3D fusion |
| `right_lidar_yolo_fusion` | Right camera + LiDAR 3D fusion |
| `multi_camera_fusion` | Combines left/right 3D detections |
| `trajectory_controller` | Executes predefined robot motion and logs data |
| `fusion_video_recorder` | Records annotated side-by-side video |

---

## Important Topics

### Camera and LiDAR

```text
/front_stereo_camera/left/image_raw
/front_stereo_camera/left/camera_info
/front_stereo_camera/right/image_raw
/front_stereo_camera/right/camera_info
/front_3d_lidar/lidar_points
```

Both cameras were tested at:

```text
640 x 360
```

### Detection Topics

```text
/yolo/detections
/yolo/right/detections

/fusion/detections_3d
/fusion/right/detections_3d

/fusion/final/detections_3d
```

### Motion

```text
/chassis/odom
/cmd_vel
```

---

## Dependencies

Tested environment:

- Ubuntu 24.04
- ROS 2 Jazzy
- Python 3.12
- NVIDIA Isaac Sim
- Nova Carter ROS-ready asset

Main Python packages:

```text
numpy==1.26.4
opencv-python
ultralytics
torch
PyYAML
```

Main ROS dependencies:

```text
rclpy
sensor_msgs
sensor_msgs_py
std_msgs
geometry_msgs
nav_msgs
vision_msgs
cv_bridge
tf2_ros
tf2_tools
```

`NumPy 1.26.4` was used because newer NumPy versions caused compatibility problems with `cv_bridge` during development.

YOLO weights are expected at:

```text
~/projects/nova_carter_yolo/models/yolov8n.pt
```

Update the path in the detector nodes if your model is stored elsewhere.

---

## Build

```bash
cd ~/ros2_ws

source /opt/ros/jazzy/setup.bash
source ~/projects/nova_carter_yolo/.venv/bin/activate

colcon build --packages-select nova_carter_perception

source install/setup.bash
```

If the project virtual environment is stored somewhere else, update the Python executable path in `setup.cfg`.

---

## Run the Complete Demo

Start Isaac Sim, load the Nova Carter scene, enable the ROS 2 bridge, and press **Play**.

Confirm that both cameras and the LiDAR are available:

```bash
ros2 topic list | grep -E 'camera|lidar|odom|cmd_vel'
```

Then start the complete stack:

```bash
ros2 launch nova_carter_perception final_demo.launch.py
```

The launch file starts the nodes in sequence:

```text
0 s  -> Left + Right YOLO
2 s  -> Left + Right LiDAR fusion
4 s  -> Multi-camera fusion
5 s  -> Video recorder
7 s  -> Trajectory controller
```

---

## Predefined Motion

The current test trajectory is relative to the robot's initial pose:

```text
0.5 m forward
0.5 m left
1.0 m backward
```

This is used only for repeatable perception validation and is not intended as a full navigation or obstacle-avoidance system.

---

## Outputs

Experiment data is saved under:

```text
~/projects/nova_carter_yolo/outputs/final_demo/run_latest/
```

Expected files:

```text
run_latest/
├── trajectory/
│   ├── planned_path.csv
│   └── executed_path.csv
├── detections/
│   └── detections.csv
└── video/
    └── perception_motion.mp4
```

The CSV files contain robot pose, velocity, object class, confidence, XYZ position, and range.

The MP4 contains side-by-side annotated left/right perception views while the robot is moving.

---

## Current Limitations

The current implementation is functional but still has several areas for improvement:

- YOLO may occasionally report **three person detections although only two person assets exist**.
- LiDAR points are selected using rectangular YOLO ROIs, so floor/background points may sometimes be included.
- No persistent object tracking IDs are used yet.
- Left/right/camera/LiDAR messages are not fully time-synchronized.
- Multi-camera matching currently uses class + 3D distance.
- Testing was completed in a lightweight scene rather than the final factory/warehouse environment.

---

## Future Work

Recommended next improvements:

- Tune YOLO confidence and NMS thresholds to reduce duplicate person detections.
- Add 3D duplicate suppression when two detections have nearly identical XYZ positions.
- Use depth clustering or DBSCAN inside each ROI.
- Remove ground-plane/background LiDAR points.
- Add timestamp synchronization using `message_filters` / ApproximateTime.
- Add persistent object tracking IDs.
- Validate XYZ/range against Isaac Sim ground truth.
- Move the completed pipeline into the supervisor-provided factory/warehouse USD.
- Test more object classes, stronger occlusion, and realistic clutter.
- Integrate Nav2 for autonomous waypoint navigation.

---

## Quick Validation

Check the final 3D output:

```bash
ros2 topic echo /fusion/final/detections_3d --once
```

Check TF:

```bash
ros2 run tf2_ros tf2_echo front_stereo_camera_left_optical front_3d_lidar
```

```bash
ros2 run tf2_ros tf2_echo front_stereo_camera_right_optical front_3d_lidar
```

---

## Author

**Youssef Mohamed**  
Mechatronics Engineering — Robotics / Autonomous Systems
