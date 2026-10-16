from launch import LaunchDescription
from launch.actions import TimerAction

from launch_ros.actions import Node


def generate_launch_description():

    # =========================================================
    # 1. LEFT YOLO
    # =========================================================

    left_yolo = Node(
        package='nova_carter_perception',
        executable='yolo_detector',
        name='yolo_detector',
        output='screen'
    )

    # =========================================================
    # 2. RIGHT YOLO
    # =========================================================

    right_yolo = Node(
        package='nova_carter_perception',
        executable='right_yolo_detector',
        name='right_yolo_detector',
        output='screen'
    )

    # =========================================================
    # 3. LEFT CAMERA + LIDAR FUSION
    #
    # Small delay gives YOLO time to publish detections.
    # =========================================================

    left_fusion = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='nova_carter_perception',
                executable='lidar_yolo_fusion',
                name='lidar_yolo_fusion',
                output='screen'
            )
        ]
    )

    # =========================================================
    # 4. RIGHT CAMERA + LIDAR FUSION
    # =========================================================

    right_fusion = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='nova_carter_perception',
                executable='right_lidar_yolo_fusion',
                name='right_lidar_yolo_fusion',
                output='screen'
            )
        ]
    )

    # =========================================================
    # 5. MULTI-CAMERA 3D FUSION
    #
    # Wait until both independent 3D fusion nodes exist.
    # =========================================================

    multi_camera_fusion = TimerAction(
        period=4.0,
        actions=[
            Node(
                package='nova_carter_perception',
                executable='multi_camera_fusion',
                name='multi_camera_fusion',
                output='screen'
            )
        ]
    )

    # =========================================================
    # 6. VIDEO RECORDER
    #
    # Starts before robot motion so the complete trajectory
    # is recorded.
    # =========================================================

    video_recorder = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='nova_carter_perception',
                executable='fusion_video_recorder',
                name='fusion_video_recorder',
                output='screen'
            )
        ]
    )

    # =========================================================
    # 7. TRAJECTORY CONTROLLER
    #
    # Starts last so:
    #
    # Cameras        ✓
    # YOLO           ✓
    # LiDAR fusion   ✓
    # Multi fusion   ✓
    # Video recorder ✓
    #
    # are already active before robot motion begins.
    # =========================================================

    trajectory_controller = TimerAction(
        period=7.0,
        actions=[
            Node(
                package='nova_carter_perception',
                executable='trajectory_controller',
                name='trajectory_controller',
                output='screen'
            )
        ]
    )

    # =========================================================
    # Launch description
    # =========================================================

    return LaunchDescription([
        left_yolo,
        right_yolo,

        left_fusion,
        right_fusion,

        multi_camera_fusion,

        video_recorder,

        trajectory_controller,
    ])