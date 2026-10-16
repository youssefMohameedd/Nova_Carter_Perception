import os
import csv
import math
import shutil

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from vision_msgs.msg import Detection3DArray


class TrajectoryController(Node):

    def __init__(self):

        super().__init__('trajectory_controller')

        # =====================================================
        # ROS TOPICS
        # =====================================================

        self.odom_topic = '/chassis/odom'
        self.cmd_vel_topic = '/cmd_vel'

        self.detections_topic = (
            '/fusion/final/detections_3d'
        )

        # =====================================================
        # SIMPLE PREDEFINED PATH
        #
        # Coordinates are RELATIVE to robot starting pose:
        #
        # +x = forward from initial robot direction
        # +y = left from initial robot direction
        #
        # This avoids depending on where the robot starts
        # in the Isaac scene.
        # =====================================================

        self.local_waypoints = [
            (0.0, 0.0),
            (0.5, 0.0),
            (0.5, 0.5),
            (-0.5, 0.5),
        ]

        # =====================================================
        # CONTROLLER SETTINGS
        # =====================================================

        self.max_linear_speed = 0.35
        self.max_angular_speed = 0.80

        self.k_linear = 0.7
        self.k_angular = 1.5

        self.position_tolerance = 0.12

        # If heading error is larger than this,
        # rotate first before translating.
        self.rotate_first_threshold = 0.35

        # =====================================================
        # ROBOT STATE
        # =====================================================

        self.start_x = None
        self.start_y = None
        self.start_yaw = None

        self.current_x = None
        self.current_y = None
        self.current_yaw = None

        self.world_waypoints = []

        self.current_waypoint_index = 1

        self.trajectory_started = False
        self.trajectory_finished = False

        # =====================================================
        # OUTPUT DIRECTORY
        # =====================================================

        self.output_dir = os.path.expanduser(
            '~/projects/nova_carter_yolo/'
            'outputs/final_demo/run_latest'
        )

        # Clear old run so today's demo is easy to find.
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

        self.trajectory_dir = os.path.join(
            self.output_dir,
            'trajectory'
        )

        self.detections_dir = os.path.join(
            self.output_dir,
            'detections'
        )

        os.makedirs(
            self.trajectory_dir,
            exist_ok=True
        )

        os.makedirs(
            self.detections_dir,
            exist_ok=True
        )

        self.planned_path_file = os.path.join(
            self.trajectory_dir,
            'planned_path.csv'
        )

        self.executed_path_file = os.path.join(
            self.trajectory_dir,
            'executed_path.csv'
        )

        self.detections_file = os.path.join(
            self.detections_dir,
            'detections.csv'
        )

        # =====================================================
        # CREATE CSV HEADERS
        # =====================================================

        with open(
            self.executed_path_file,
            'w',
            newline=''
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                'timestamp',
                'x',
                'y',
                'yaw',
                'linear_velocity',
                'angular_velocity'
            ])

        with open(
            self.detections_file,
            'w',
            newline=''
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                'timestamp',
                'detection_index',
                'class',
                'confidence',
                'x',
                'y',
                'z',
                'range'
            ])

        # =====================================================
        # ROS
        # =====================================================

        self.cmd_pub = self.create_publisher(
            Twist,
            self.cmd_vel_topic,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            qos_profile_sensor_data
        )

        self.detection_sub = self.create_subscription(
            Detection3DArray,
            self.detections_topic,
            self.detection_callback,
            10
        )

        # Control loop at 10 Hz
        self.timer = self.create_timer(
            0.1,
            self.control_loop
        )

        self.get_logger().info(
            'Trajectory controller started'
        )

        self.get_logger().info(
            f'Odom: {self.odom_topic}'
        )

        self.get_logger().info(
            f'CmdVel: {self.cmd_vel_topic}'
        )

        self.get_logger().info(
            f'Detections: {self.detections_topic}'
        )

        self.get_logger().info(
            f'Output directory: {self.output_dir}'
        )

        self.get_logger().info(
            'Waiting for first odometry message...'
        )

    # =========================================================
    # MATH HELPERS
    # =========================================================

    def normalize_angle(self, angle):

        while angle > math.pi:
            angle -= 2.0 * math.pi

        while angle < -math.pi:
            angle += 2.0 * math.pi

        return angle

    def quaternion_to_yaw(self, q):

        siny_cosp = (
            2.0
            * (
                q.w * q.z
                + q.x * q.y
            )
        )

        cosy_cosp = (
            1.0
            - 2.0
            * (
                q.y * q.y
                + q.z * q.z
            )
        )

        return math.atan2(
            siny_cosp,
            cosy_cosp
        )

    def timestamp_to_seconds(self, stamp):

        return (
            float(stamp.sec)
            + float(stamp.nanosec) * 1e-9
        )

    # =========================================================
    # INITIALIZE WORLD WAYPOINTS
    # =========================================================

    def initialize_waypoints(self):

        self.world_waypoints = []

        cos_yaw = math.cos(
            self.start_yaw
        )

        sin_yaw = math.sin(
            self.start_yaw
        )

        for local_x, local_y in self.local_waypoints:

            world_x = (
                self.start_x
                + local_x * cos_yaw
                - local_y * sin_yaw
            )

            world_y = (
                self.start_y
                + local_x * sin_yaw
                + local_y * cos_yaw
            )

            self.world_waypoints.append(
                (
                    world_x,
                    world_y
                )
            )

        # Save planned path
        with open(
            self.planned_path_file,
            'w',
            newline=''
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                'waypoint',
                'x',
                'y'
            ])

            for index, waypoint in enumerate(
                self.world_waypoints
            ):

                writer.writerow([
                    index,
                    waypoint[0],
                    waypoint[1]
                ])

        self.get_logger().info(
            'Predefined trajectory initialized'
        )

        for index, (
            x,
            y
        ) in enumerate(self.world_waypoints):

            self.get_logger().info(
                f'Waypoint {index}: '
                f'({x:.2f}, {y:.2f})'
            )

        self.get_logger().info(
            f'Planned path saved: '
            f'{self.planned_path_file}'
        )

    # =========================================================
    # ODOM CALLBACK
    # =========================================================

    def odom_callback(self, msg):

        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        self.current_x = float(
            position.x
        )

        self.current_y = float(
            position.y
        )

        self.current_yaw = (
            self.quaternion_to_yaw(
                orientation
            )
        )

        # =====================================================
        # First odom message defines start pose
        # =====================================================

        if self.start_x is None:

            self.start_x = self.current_x
            self.start_y = self.current_y
            self.start_yaw = self.current_yaw

            self.initialize_waypoints()

            self.trajectory_started = True

            self.get_logger().info(
                'Trajectory STARTED'
            )

        # =====================================================
        # Log actual motion
        # =====================================================

        timestamp = self.timestamp_to_seconds(
            msg.header.stamp
        )

        linear_velocity = float(
            msg.twist.twist.linear.x
        )

        angular_velocity = float(
            msg.twist.twist.angular.z
        )

        with open(
            self.executed_path_file,
            'a',
            newline=''
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                timestamp,
                self.current_x,
                self.current_y,
                self.current_yaw,
                linear_velocity,
                angular_velocity
            ])

    # =========================================================
    # FINAL DETECTIONS LOGGER
    # =========================================================

    def detection_callback(self, msg):

        timestamp = self.timestamp_to_seconds(
            msg.header.stamp
        )

        with open(
            self.detections_file,
            'a',
            newline=''
        ) as file:

            writer = csv.writer(file)

            for index, detection in enumerate(
                msg.detections
            ):

                class_name = 'unknown'
                confidence = 0.0

                if len(detection.results) > 0:

                    hypothesis = (
                        detection.results[0]
                        .hypothesis
                    )

                    class_name = (
                        hypothesis.class_id
                    )

                    confidence = float(
                        hypothesis.score
                    )

                x = float(
                    detection.bbox
                    .center.position.x
                )

                y = float(
                    detection.bbox
                    .center.position.y
                )

                z = float(
                    detection.bbox
                    .center.position.z
                )

                range_value = math.sqrt(
                    x*x + y*y + z*z
                )

                writer.writerow([
                    timestamp,
                    index + 1,
                    class_name,
                    confidence,
                    x,
                    y,
                    z,
                    range_value
                ])

    # =========================================================
    # STOP ROBOT
    # =========================================================

    def stop_robot(self):

        cmd = Twist()

        cmd.linear.x = 0.0
        cmd.angular.z = 0.0

        self.cmd_pub.publish(
            cmd
        )

    # =========================================================
    # CONTROL LOOP
    # =========================================================

    def control_loop(self):

        if not self.trajectory_started:
            return

        if self.trajectory_finished:

            self.stop_robot()
            return

        if self.current_x is None:
            return

        # =====================================================
        # Completed all waypoints
        # =====================================================

        if (
            self.current_waypoint_index
            >= len(self.world_waypoints)
        ):

            self.trajectory_finished = True

            self.stop_robot()

            self.get_logger().info(
                '===================================='
            )

            self.get_logger().info(
                'TRAJECTORY COMPLETED'
            )

            self.get_logger().info(
                f'Executed path: '
                f'{self.executed_path_file}'
            )

            self.get_logger().info(
                f'Detections: '
                f'{self.detections_file}'
            )

            self.get_logger().info(
                '===================================='
            )

            return

        # =====================================================
        # Current target
        # =====================================================

        target_x, target_y = (
            self.world_waypoints[
                self.current_waypoint_index
            ]
        )

        dx = (
            target_x
            - self.current_x
        )

        dy = (
            target_y
            - self.current_y
        )

        distance = math.sqrt(
            dx*dx + dy*dy
        )

        # =====================================================
        # Waypoint reached
        # =====================================================

        if distance < self.position_tolerance:

            self.get_logger().info(
                f'Waypoint '
                f'{self.current_waypoint_index} '
                f'REACHED'
            )

            self.current_waypoint_index += 1

            self.stop_robot()

            return

        # =====================================================
        # Required heading
        # =====================================================

        target_heading = math.atan2(
            dy,
            dx
        )

        heading_error = (
            self.normalize_angle(
                target_heading
                - self.current_yaw
            )
        )

        # =====================================================
        # Controller
        # =====================================================

        cmd = Twist()

        # Rotate first if target is far off-axis.
        if (
            abs(heading_error)
            > self.rotate_first_threshold
        ):

            cmd.linear.x = 0.0

            angular = (
                self.k_angular
                * heading_error
            )

        else:

            linear = (
                self.k_linear
                * distance
            )

            cmd.linear.x = min(
                linear,
                self.max_linear_speed
            )

            angular = (
                self.k_angular
                * heading_error
            )

        # Clamp angular velocity
        angular = max(
            -self.max_angular_speed,
            min(
                self.max_angular_speed,
                angular
            )
        )

        cmd.angular.z = angular

        self.cmd_pub.publish(
            cmd
        )

    # =========================================================
    # SHUTDOWN
    # =========================================================

    def destroy_node(self):

        self.stop_robot()

        self.get_logger().info(
            'Robot stopped'
        )

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = TrajectoryController()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()