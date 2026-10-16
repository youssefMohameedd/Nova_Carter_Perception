import os

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class FusionVideoRecorder(Node):

    def __init__(self):

        super().__init__('fusion_video_recorder')

        # =====================================================
        # Topics
        # =====================================================

        self.left_topic = (
            '/fusion/debug_image'
        )

        self.right_topic = (
            '/fusion/right/debug_image'
        )

        # =====================================================
        # Output
        # =====================================================

        self.output_dir = os.path.expanduser(
            '~/projects/nova_carter_yolo/'
            'outputs/final_demo/run_latest/video'
        )

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        self.video_path = os.path.join(
            self.output_dir,
            'perception_motion.mp4'
        )

        # =====================================================
        # Video settings
        # =====================================================

        self.fps = 10.0

        self.bridge = CvBridge()

        self.latest_left = None
        self.latest_right = None

        self.writer = None

        self.frame_count = 0

        # =====================================================
        # ROS
        # =====================================================

        self.left_sub = self.create_subscription(
            Image,
            self.left_topic,
            self.left_callback,
            10
        )

        self.right_sub = self.create_subscription(
            Image,
            self.right_topic,
            self.right_callback,
            10
        )

        # Write approximately at 10 FPS
        self.timer = self.create_timer(
            0.1,
            self.write_frame
        )

        self.get_logger().info(
            'Fusion video recorder started'
        )

        self.get_logger().info(
            f'LEFT: {self.left_topic}'
        )

        self.get_logger().info(
            f'RIGHT: {self.right_topic}'
        )

        self.get_logger().info(
            f'Video: {self.video_path}'
        )

    # =========================================================
    # Camera callbacks
    # =========================================================

    def left_callback(self, msg):

        try:

            self.latest_left = (
                self.bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding='bgr8'
                )
            )

        except Exception as ex:

            self.get_logger().error(
                f'Left cv_bridge error: {ex}'
            )

    def right_callback(self, msg):

        try:

            self.latest_right = (
                self.bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding='bgr8'
                )
            )

        except Exception as ex:

            self.get_logger().error(
                f'Right cv_bridge error: {ex}'
            )

    # =========================================================
    # Initialize VideoWriter
    # =========================================================

    def initialize_writer(
        self,
        width,
        height
    ):

        fourcc = cv2.VideoWriter_fourcc(
            *'mp4v'
        )

        self.writer = cv2.VideoWriter(
            self.video_path,
            fourcc,
            self.fps,
            (
                width,
                height
            )
        )

        if not self.writer.isOpened():

            self.get_logger().error(
                'Could not open VideoWriter'
            )

            self.writer = None

            return False

        self.get_logger().info(
            f'Video recording started | '
            f'{width}x{height} @ '
            f'{self.fps:.1f} FPS'
        )

        return True

    # =========================================================
    # Main recording loop
    # =========================================================

    def write_frame(self):

        if self.latest_left is None:
            return

        if self.latest_right is None:
            return

        left = self.latest_left.copy()
        right = self.latest_right.copy()

        # =====================================================
        # Make both images same size
        # =====================================================

        height = min(
            left.shape[0],
            right.shape[0]
        )

        width = min(
            left.shape[1],
            right.shape[1]
        )

        left = cv2.resize(
            left,
            (
                width,
                height
            )
        )

        right = cv2.resize(
            right,
            (
                width,
                height
            )
        )

        # =====================================================
        # Add camera labels
        # =====================================================

        cv2.putText(
            left,
            'LEFT CAMERA',
            (10, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            right,
            'RIGHT CAMERA',
            (10, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # =====================================================
        # Side-by-side frame
        # =====================================================

        combined = np.hstack(
            (
                left,
                right
            )
        )

        # =====================================================
        # Add project title
        # =====================================================

        cv2.putText(
            combined,
            'Dual-Camera + 3D LiDAR Perception',
            (combined.shape[1] // 2 - 185, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # =====================================================
        # Start writer when dimensions known
        # =====================================================

        if self.writer is None:

            success = self.initialize_writer(
                combined.shape[1],
                combined.shape[0]
            )

            if not success:
                return

        self.writer.write(
            combined
        )

        self.frame_count += 1

        if (
            self.frame_count
            % 50
            == 0
        ):

            self.get_logger().info(
                f'Recorded '
                f'{self.frame_count} frames'
            )

    # =========================================================
    # Shutdown
    # =========================================================

    def destroy_node(self):

        if self.writer is not None:

            self.writer.release()

            self.get_logger().info(
                f'Video saved: '
                f'{self.video_path}'
            )

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = FusionVideoRecorder()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()