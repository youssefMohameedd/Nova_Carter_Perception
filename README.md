# Nova Carter Perception

**Dual-Camera YOLO + 3D LiDAR Perception and ROS 2 Sensor Fusion in NVIDIA Isaac Sim**

This repository contains a ROS 2 perception pipeline developed for the **Nova Carter mobile robot** in **NVIDIA Isaac Sim**. The project combines two front RGB cameras, a 3D LiDAR, YOLO object detection, ROS 2 TF transforms, camera–LiDAR projection, multi-camera association, predefined robot motion, and experiment logging.

The final pipeline converts independent 2D detections from the left and right cameras into robot-centric **3D object detections** published using standard ROS 2 `vision_msgs` messages.

Repository:  
https://github.com/youssefMohameedd/Nova_Carter_Perception

---

## 1. Project Objectives

The main objectives of this work are to:

- Stream Nova Carter RGB camera and 3D LiDAR data from Isaac Sim into ROS 2.
- Perform real-time object detection using **YOLOv8n**.
- Project 3D LiDAR points into each camera image using camera calibration and ROS TF.
- Associate LiDAR points with YOLO 2D bounding boxes.
- Estimate object **XYZ position and range**.
- Publish object results as standard `vision_msgs/msg/Detection3DArray`.
- Add a second front camera for multi-view redundancy.
- Associate left- and right-camera 3D observations in the common `base_link` frame.
- Continue perception while the robot moves through a predefined trajectory.
- Export trajectory data, object detections, annotated images, and a side-by-side perception video.
- Provide a repeatable end-to-end experiment through a ROS 2 launch file.

---

## 2. Final System Architecture

```text
                    NVIDIA ISAAC SIM
                           |
        +------------------+------------------+
        |                  |                  |
        v                  v                  v
   Left RGB Camera    Right RGB Camera      3D LiDAR
        |                  |                  |
        v                  v                  |
    YOLOv8n            YOLOv8n                |
        |                  |                  |
 Detection2DArray    Detection2DArray         |
        |                  |                  |
        +---------+--------+------------------+
                  |        |
                  v        v
         Left Camera-LiDAR Fusion
         Right Camera-LiDAR Fusion
                  |        |
                  v        v
           Detection3DArray
                  \        /
                   \      /
                    v    v
              Multi-Camera Fusion
                      |
                      v
          /fusion/final/detections_3d
                      |
             vision_msgs/Detection3DArray

                      +
                      |
              Trajectory Controller
             /chassis/odom -> /cmd_vel
                      |
                      v
                Robot Motion

                      +
                      |
             Experiment Recording
        CSV trajectory + detections + MP4
```

---

## 3. Research and Development Progression

The project was developed incrementally so that every stage could be independently validated before adding more complexity.

### Stage 1 — ROS 2 / Isaac Sim Communication

The first objective was to establish reliable ROS 2 communication between Isaac Sim and the Nova Carter sensors.

Validated streams included:

- Front RGB camera
- Camera calibration information
- 3D LiDAR
- IMU
- Robot odometry
- `/cmd_vel`
- `/tf`
- `/tf_static`

### Stage 2 — RGB Camera Validation

A basic ROS 2 image listener was created to confirm that RGB frames could be received and converted through `cv_bridge`.

The initial left-camera stream is:

```text
/front_stereo_camera/left/image_raw
```

and its calibration data is published on:

```text
/front_stereo_camera/left/camera_info
```

### Stage 3 — YOLO Object Detection

YOLOv8n was integrated into a ROS 2 node.

The detector publishes:

```text
/yolo/detections
/yolo/annotated_image
```

with detections represented using:

```text
vision_msgs/msg/Detection2DArray
```

A second YOLO node was later added for the right front camera:

```text
/yolo/right/detections
/yolo/right/annotated_image
```

The project was tested primarily with person assets because they are directly supported by the pretrained COCO classes.

### Stage 4 — TF-Based Sensor Alignment

Camera and LiDAR measurements exist in different coordinate frames. ROS TF was therefore used to transform LiDAR points into each camera optical frame before projection.

Main frames:

```text
base_link
front_3d_lidar
front_stereo_camera_left_optical
front_stereo_camera_right_optical
```

A useful geometric sanity check obtained during development was:

```text
LiDAR -> LEFT camera  : x translation ~ +0.075 m
LiDAR -> RIGHT camera : x translation ~ -0.075 m
```

The symmetric ±75 mm offsets are consistent with the mirrored left/right sensor placement around the LiDAR.

### Stage 5 — LiDAR-to-Camera Projection

Each LiDAR point is transformed to camera coordinates:

```text
P_camera = R * P_lidar + t
```

Then projected to image pixels using camera intrinsics:

```text
u = fx * X / Z + cx
v = fy * Y / Z + cy
```

Only points:

- in front of the camera, and
- inside the camera image boundaries

are retained.

The intermediate projected cloud can be inspected through:

```text
/fusion/camera_fov_points
```

### Stage 6 — YOLO + LiDAR Object Fusion

For each YOLO bounding box, projected LiDAR points inside the 2D ROI are selected.

The current implementation estimates the object position using a median of the associated LiDAR points and transforms the result to:

```text
base_link
```

The left fusion node publishes:

```text
/fusion/detections_3d
```

and the right fusion node publishes:

```text
/fusion/right/detections_3d
```

Both use:

```text
vision_msgs/msg/Detection3DArray
```

### Stage 7 — Dual-Camera Fusion

Both independent camera-LiDAR pipelines express their final 3D detections in the common `base_link` frame.

The multi-camera fusion node associates detections using:

1. Same object class.
2. Small 3D Euclidean separation.
3. Closest valid match.

Matched observations are merged, while unmatched observations from either camera are preserved.

This is important because a person may temporarily be detected by one camera but missed by the other.

The final perception output is:

```text
/fusion/final/detections_3d
```

### Stage 8 — Dynamic Validation

A simple ROS 2 trajectory controller was added to validate perception while the robot moves.

The controller subscribes to:

```text
/chassis/odom
```

and publishes velocity commands to:

```text
/cmd_vel
```

The current test path is relative to the robot starting pose:

```text
Start
 -> 0.5 m forward
 -> 0.5 m left
 -> 1.0 m backward
```

This is intended as a **repeatable perception-validation trajectory**, not a complete autonomous navigation solution.

### Stage 9 — Experiment Logging and Video Export

The system records:

- Planned trajectory
- Executed odometry
- Fused detections
- Annotated side-by-side perception video

The video combines the left and right fusion views while the robot is moving.

---

## 4. Main ROS 2 Nodes

| Node / Executable | Function |
|---|---|
| `rgb_listener` | Basic RGB stream validation |
| `yolo_detector` | YOLO inference on the left camera |
| `right_yolo_detector` | YOLO inference on the right camera |
| `lidar_camera_projection` | Standalone LiDAR-to-camera projection/debugging |
| `lidar_yolo_fusion` | Left camera + LiDAR 3D fusion |
| `right_lidar_yolo_fusion` | Right camera + LiDAR 3D fusion |
| `multi_camera_fusion` | Associates and combines left/right 3D detections |
| `trajectory_controller` | Executes the predefined robot trajectory and logs data |
| `fusion_video_recorder` | Records left/right annotated fusion video |

---

## 5. Important ROS Topics

### Isaac Sim inputs

```text
/front_stereo_camera/left/image_raw
/front_stereo_camera/left/camera_info

/front_stereo_camera/right/image_raw
/front_stereo_camera/right/camera_info

/front_3d_lidar/lidar_points

/chassis/odom
/cmd_vel

/tf
/tf_static
```

### YOLO outputs

```text
/yolo/detections
/yolo/annotated_image

/yolo/right/detections
/yolo/right/annotated_image
```

### Left camera-LiDAR fusion

```text
/fusion/camera_fov_points
/fusion/object_points
/fusion/debug_image
/fusion/detections_3d
```

### Right camera-LiDAR fusion

```text
/fusion/right/camera_fov_points
/fusion/right/object_points
/fusion/right/debug_image
/fusion/right/detections_3d
```

### Final multi-camera output

```text
/fusion/final/detections_3d
```

---

## 6. Development Environment

The project was developed and tested with approximately the following environment:

### Operating system

- Ubuntu 24.04 LTS

### ROS

- ROS 2 Jazzy

### Simulation

- NVIDIA Isaac Sim
- Nova Carter ROS-ready robot asset
- Front Hawk stereo cameras
- Front 3D LiDAR
- ROS 2 Bridge enabled

The local development work was performed with Isaac Sim 5.x. Newer Isaac Sim releases may use slightly different Action Graph node layouts or ROS helper nodes.

### Python

- Python 3.12
- Python virtual environment

### Main Python packages

- `rclpy`
- `numpy`
- `opencv-python`
- `cv_bridge`
- `ultralytics`
- `torch`
- `PyYAML`
- `sensor_msgs_py`

### Main ROS message / TF packages

- `sensor_msgs`
- `std_msgs`
- `geometry_msgs`
- `nav_msgs`
- `vision_msgs`
- `tf2_ros`
- `tf2_tools`

### Known working package versions during development

The exact versions can vary by machine, but the project was successfully tested with:

```text
NumPy      1.26.4
OpenCV     4.10.x
Python     3.12
ROS 2      Jazzy
```

A CUDA-capable PyTorch installation can be used if GPU inference is desired. During development, YOLO was also successfully executed on CPU.

> **Important:** NumPy 1.26.4 was used to avoid compatibility problems encountered between newer NumPy versions and ROS `cv_bridge`.

---

## 7. Isaac Sim Configuration

Before running the ROS package, the Nova Carter simulation must already be running.

The launch file **does not start Isaac Sim**.

### Required sensor configuration

The simulation should publish both front cameras at:

```text
Width  = 640
Height = 360
```

Expected camera frames:

```text
front_stereo_camera_left_optical
front_stereo_camera_right_optical
```

Expected LiDAR frame:

```text
front_3d_lidar
```

The TF tree must connect the sensors to:

```text
base_link
```

### Right Camera

In the Front Hawk Action Graph, make sure the right-camera:

- `Isaac Create Render Product`
- ROS 2 camera publishing helper
- CameraInfo publishing path

are enabled.

Verify with:

```bash
ros2 topic list | grep front_stereo_camera
```

Expected:

```text
/front_stereo_camera/left/camera_info
/front_stereo_camera/left/image_raw
/front_stereo_camera/right/camera_info
/front_stereo_camera/right/image_raw
```

---

## 8. Installation

### 8.1 Clone the repository

```bash
cd ~/ros2_ws/src

git clone git@github.com:youssefMohameedd/Nova_Carter_Perception.git
```

If cloning the repository as a complete workspace rather than only the ROS package, place the package so that the final structure contains:

```text
~/ros2_ws/src/nova_carter_perception/
```

### 8.2 Source ROS 2

```bash
source /opt/ros/jazzy/setup.bash
```

### 8.3 Create / activate a Python environment

A Python virtual environment is recommended.

Example:

```bash
python3 -m venv ~/projects/nova_carter_yolo/.venv --system-site-packages

source ~/projects/nova_carter_yolo/.venv/bin/activate
```

Install Python dependencies:

```bash
python -m pip install --upgrade pip setuptools

pip install \
    numpy==1.26.4 \
    opencv-python \
    ultralytics \
    PyYAML
```

Install a PyTorch build suitable for your CPU/CUDA configuration following the official PyTorch installation instructions.

ROS Python dependencies such as `rclpy`, `cv_bridge`, `vision_msgs`, `sensor_msgs`, `nav_msgs`, and TF2 should be installed through the ROS 2 Jazzy packages.

### 8.4 YOLO model

The detector nodes expect YOLOv8n weights.

The development environment used:

```text
~/projects/nova_carter_yolo/models/yolov8n.pt
```

Either place the model there or edit the model path inside the YOLO detector nodes.

### 8.5 Important virtual-environment note

During development, ROS console scripts were configured to use the project virtual-environment Python interpreter.

If `setup.cfg` contains a hard-coded Python path such as:

```text
/home/youssef/projects/nova_carter_yolo/.venv/bin/python3
```

change it to the corresponding path on your own machine before building.

This is especially important if `ros2 run` can import ROS packages but cannot import `ultralytics`.

---

## 9. Build

From the ROS workspace:

```bash
cd ~/ros2_ws

source /opt/ros/jazzy/setup.bash
source ~/projects/nova_carter_yolo/.venv/bin/activate

colcon build --packages-select nova_carter_perception

source install/setup.bash
```

After every Python-node or `setup.py` modification, rebuild the package.

---

## 10. Running the Complete Demo

### 10.1 Start Isaac Sim

1. Load the Nova Carter scene.
2. Confirm that both cameras, LiDAR, TF, odometry, and `/cmd_vel` are active.
3. Press **Play**.

Quick check:

```bash
ros2 topic list | grep -E 'camera|lidar|odom|cmd_vel|tf'
```

### 10.2 Start the full ROS pipeline

```bash
cd ~/ros2_ws

source /opt/ros/jazzy/setup.bash
source ~/projects/nova_carter_yolo/.venv/bin/activate
source install/setup.bash

ros2 launch nova_carter_perception final_demo.launch.py
```

The launch file starts the system in stages so that upstream perception topics are available before downstream nodes begin:

```text
~0 s : Left YOLO + Right YOLO
~2 s : Left fusion + Right fusion
~4 s : Multi-camera fusion
~5 s : Video recorder
~7 s : Trajectory controller
```

The robot then begins the predefined motion while perception and logging continue.

---

## 11. Running Nodes Manually

For debugging, nodes can also be started separately.

### Left YOLO

```bash
ros2 run nova_carter_perception yolo_detector
```

### Right YOLO

```bash
ros2 run nova_carter_perception right_yolo_detector
```

### Left camera + LiDAR fusion

```bash
ros2 run nova_carter_perception lidar_yolo_fusion
```

### Right camera + LiDAR fusion

```bash
ros2 run nova_carter_perception right_lidar_yolo_fusion
```

### Multi-camera fusion

```bash
ros2 run nova_carter_perception multi_camera_fusion
```

### Trajectory controller

```bash
ros2 run nova_carter_perception trajectory_controller
```

### Video recorder

```bash
ros2 run nova_carter_perception fusion_video_recorder
```

---

## 12. Validation Commands

### Verify the final 3D perception output

```bash
ros2 topic type /fusion/final/detections_3d
```

Expected:

```text
vision_msgs/msg/Detection3DArray
```

Inspect one message:

```bash
ros2 topic echo /fusion/final/detections_3d --once
```

### Verify camera rates

```bash
ros2 topic hz /front_stereo_camera/left/image_raw
ros2 topic hz /front_stereo_camera/right/image_raw
```

### Verify LiDAR rate

```bash
ros2 topic hz /front_3d_lidar/lidar_points
```

### Verify TF

LiDAR to left camera:

```bash
ros2 run tf2_ros tf2_echo \
front_stereo_camera_left_optical \
front_3d_lidar
```

LiDAR to right camera:

```bash
ros2 run tf2_ros tf2_echo \
front_stereo_camera_right_optical \
front_3d_lidar
```

Right camera to robot base:

```bash
ros2 run tf2_ros tf2_echo \
base_link \
front_stereo_camera_right_optical
```

---

## 13. Experiment Outputs

The trajectory controller and video recorder create experiment artifacts under:

```text
~/projects/nova_carter_yolo/outputs/final_demo/run_latest/
```

Expected structure:

```text
run_latest/
├── trajectory/
│   ├── planned_path.csv
│   └── executed_path.csv
│
├── detections/
│   └── detections.csv
│
└── video/
    └── perception_motion.mp4
```

### `planned_path.csv`

Contains predefined trajectory waypoints.

### `executed_path.csv`

Contains:

```text
timestamp
x
y
yaw
linear_velocity
angular_velocity
```

### `detections.csv`

Contains:

```text
timestamp
detection_index
class
confidence
x
y
z
range
```

### `perception_motion.mp4`

Side-by-side annotated video of the left and right fusion streams during robot motion.

The video includes:

- Object bounding boxes
- YOLO class/confidence
- Camera-FOV LiDAR points
- Object-associated LiDAR points
- Estimated object position/range
- Left/right camera visualization

---

## 14. Current Results

The implemented system successfully demonstrates:

- Live RGB streaming from both front cameras.
- YOLOv8n inference for both viewpoints.
- Multiple simultaneous person detections.
- 3D LiDAR projection into both RGB images.
- TF-based sensor coordinate transformation.
- ROI-based LiDAR association.
- Robot-centric XYZ and range estimation.
- Standard ROS `Detection3DArray` output.
- Left/right 3D detection association.
- Preservation of single-camera observations when the other camera temporarily misses an object.
- Dynamic validation while Nova Carter moves.
- Automated CSV logging.
- Annotated video export.
- One-command ROS 2 launch of the complete perception stack.

---

## 15. Known Limitations

### 15.1 Duplicate / noisy person detections

During testing, the simulated environment sometimes contained **two person assets while YOLO produced three person detections**.

The extra result is generally caused by duplicate or unstable 2D detections rather than the LiDAR creating a new physical object.

Current mitigation includes:

- Higher YOLO confidence threshold.
- YOLO Non-Maximum Suppression (NMS) IoU configuration.

However, robust object identity filtering is not implemented yet.

### 15.2 ROI-based LiDAR association

The current method considers LiDAR points whose image projections fall inside a YOLO bounding box.

A rectangular 2D ROI may also contain:

- floor points,
- walls,
- background,
- neighboring objects,
- overlapping-object points.

Therefore the median estimate can still contain depth noise.

### 15.3 No persistent object tracking

The displayed labels:

```text
person #1
person #2
```

are frame-local indexes, not persistent tracking IDs.

### 15.4 Sensor timing

The current fusion implementation primarily uses the most recently received camera, detection, and LiDAR information.

For static or slow tests this is acceptable, but accurate fusion during faster motion should use explicit timestamp synchronization.

### 15.5 Environment

The software was finalized and validated in a lightweight controlled Isaac Sim scene to reduce GPU load and development/debugging time.

Deployment in a more complex factory/warehouse USD is planned as future work.

---

## 16. Future Work

### A. Eliminate duplicate detections when only two physical people exist

A high-priority improvement is to ensure that if the simulation contains two people, the final output converges to two persistent physical objects rather than occasionally reporting three detections.

Possible improvements:

- Tune YOLO confidence and NMS thresholds.
- Apply class-wise 2D duplicate suppression.
- Apply 3D spatial duplicate suppression in `base_link`.
- Merge same-class detections whose XYZ estimates are nearly identical.
- Add temporal filtering so detections must remain consistent over multiple frames.
- Introduce persistent object IDs and multi-object tracking.
- Use confidence history instead of a single-frame confidence value.

### B. Better LiDAR object-point filtering

Replace raw rectangular ROI selection with:

- ROI shrinking
- Depth histograms
- Nearest coherent depth cluster
- DBSCAN / Euclidean clustering
- Statistical outlier removal
- Ground-plane removal
- Object segmentation masks instead of only bounding boxes

This should significantly improve object range estimation in crowded scenes.

### C. Timestamp synchronization

Use ROS 2 message synchronization so camera frames, YOLO detections, and LiDAR scans correspond to the same acquisition time.

Possible approaches include:

- `message_filters`
- ApproximateTime synchronization
- timestamp-aware detection buffering

This becomes particularly important when the robot is moving.

### D. Persistent multi-object tracking

Add tracking across frames using:

- nearest-neighbor 3D tracking,
- Kalman filtering,
- ByteTrack / BoT-SORT,
- 3D association,
- persistent ROS object IDs.

### E. Quantitative ground-truth validation

Compare predicted object distance against Isaac Sim ground truth.

Metrics should include:

- Mean Absolute Error (MAE)
- Range error
- XYZ error
- Valid 3D detection rate
- Detection continuity
- Processing latency
- FPS

### F. Factory / Warehouse deployment

Transfer the completed ROS perception architecture into the supervisor-provided factory/warehouse USD environment.

Validation should include:

- More realistic clutter
- Multiple object classes
- Stronger occlusion
- Dynamic viewpoints
- Lighting variations
- More distant targets
- Multiple workers
- Shelves and industrial objects

### G. Autonomous navigation

The current trajectory controller is a repeatable motion test rather than a planner.

Future work can integrate:

- Nav2
- Occupancy maps
- Waypoint navigation
- Obstacle avoidance
- Perception-aware path planning

### H. GPU acceleration and performance optimization

The current project can be extended with:

- CUDA YOLO inference
- Reduced-copy image processing
- lower-resolution debug visualization
- decoupled inference/publishing rates
- optimized LiDAR clustering
- lifecycle nodes / composable nodes

---

## 17. Troubleshooting

### Fusion node starts but produces no output

The fusion node requires YOLO detections.

If running nodes manually, start the YOLO detector first:

```bash
ros2 run nova_carter_perception yolo_detector
```

The launch file handles this startup sequence automatically.

### Right camera topics do not exist

Verify that the right Front Hawk render-product and ROS camera publisher are enabled in Isaac Sim.

Expected topics:

```text
/front_stereo_camera/right/image_raw
/front_stereo_camera/right/camera_info
```

### `TF unavailable`

Check:

```bash
ros2 topic echo /tf_static --once
```

and:

```bash
ros2 run tf2_ros tf2_echo \
front_stereo_camera_right_optical \
front_3d_lidar
```

A short `Invalid frame ID` message immediately after node startup can occur before TF has been received. Persistent failure indicates a missing frame or TF publisher.

### `ultralytics` cannot be imported with `ros2 run`

The ROS console-script interpreter may not be the same Python interpreter as the project virtual environment.

Check the executable configuration in `setup.cfg` and make sure it points to the Python environment containing Ultralytics.

### Video file is incomplete

Stop the launch cleanly using:

```text
Ctrl+C
```

The recorder must release the OpenCV `VideoWriter` to finalize the MP4 file.

---

## 18. Suggested Repository Cleanup

ROS workspaces generate large files that should normally not be committed.

A recommended `.gitignore` is:

```gitignore
# ROS 2 generated directories
build/
install/
log/

# Python
__pycache__/
*.py[cod]
*.egg-info/

# Virtual environments
.venv/
venv/
env/

# Experiment outputs
outputs/
*.mp4
*.avi

# ROS bags
*.db3
*.mcap
rosbag2_*/

# IDE
.vscode/
.idea/

# Temporary files
*.tmp
*.log
```

Only source code, launch files, package configuration, documentation, and useful small project assets should normally be kept under version control.

---

## 19. Repository Structure

A cleaned source tree is expected to look approximately like:

```text
Nova_Carter_Perception/
├── src/
│   └── nova_carter_perception/
│       ├── launch/
│       │   └── final_demo.launch.py
│       │
│       ├── nova_carter_perception/
│       │   ├── __init__.py
│       │   ├── rgb_listener.py
│       │   ├── yolo_detector.py
│       │   ├── yolo_detector_right.py
│       │   ├── lidar_camera_projection.py
│       │   ├── lidar_yolo_fusion.py
│       │   ├── right_lidar_yolo_fusion.py
│       │   ├── multi_camera_fusion.py
│       │   ├── trajectory_controller.py
│       │   └── fusion_video_recorder.py
│       │
│       ├── package.xml
│       ├── setup.py
│       └── setup.cfg
│
├── README.md
├── .gitignore
└── LICENSE
```

---

## 20. Final Pipeline Summary

```text
Isaac Sim
   |
   +--> Left RGB --> YOLO -----------+
   |                                 |
   +--> Right RGB --> YOLO ----------+--> Camera-LiDAR Fusion
   |                                 |          |
   +--> 3D LiDAR --------------------+          v
   |                                      Detection3DArray
   |                                             |
   |                                    Multi-Camera Fusion
   |                                             |
   |                                             v
   |                              /fusion/final/detections_3d
   |
   +--> Odometry --> Trajectory Controller --> /cmd_vel

Final outputs:
- Standardized ROS 3D detections
- Planned / executed trajectory CSV
- Detection CSV
- Annotated dual-camera MP4
```

The project demonstrates a complete experimental perception workflow from **simulated sensor acquisition** through **deep-learning detection**, **geometric sensor fusion**, **multi-camera association**, **robot motion**, and **data export**.

---

## License

This repository contains an MIT license in the project root.

## Author

**Youssef Mohamed**  
Mechatronics Engineering / Robotics and Autonomous Systems
