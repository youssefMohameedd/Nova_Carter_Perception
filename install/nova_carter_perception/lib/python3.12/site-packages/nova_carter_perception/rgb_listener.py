import os

import cv2
import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class RGBListener(Node):

    def __init__(self):
        super().__init__('rgb_listener')

        self.bridge = CvBridge()

        self.topic = '/front_stereo_camera/left/image_raw'

        self.subscription = self.create_subscription(
            Image,
            self.topic,
            self.image_callback,
            qos_profile_sensor_data
        )

        self.frame_count = 0
        self.first_frame_saved = False

        self.output_path = os.path.expanduser(
            '~/projects/nova_carter_yolo/outputs/debug/Presentation_frame.jpg'
        )

        os.makedirs(
            os.path.dirname(self.output_path),
            exist_ok=True
        )

        self.get_logger().info(
            f'Listening to {self.topic}'
        )


    def image_callback(self, msg):

        self.frame_count += 1

        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

        except Exception as error:

            self.get_logger().error(
                f'Image conversion failed: {error}'
            )

            return

        height, width = frame.shape[:2]

        if self.frame_count % 30 == 0:

            self.get_logger().info(
                f'Frame {self.frame_count}: '
                f'{width}x{height}, '
                f'encoding={msg.encoding}'
            )

        if not self.first_frame_saved:

            cv2.imwrite(
                self.output_path,
                frame
            )

            self.first_frame_saved = True

            self.get_logger().info(
                f'First RGB frame saved to: {self.output_path}'
            )


def main(args=None):

    rclpy.init(args=args)

    node = RGBListener()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()