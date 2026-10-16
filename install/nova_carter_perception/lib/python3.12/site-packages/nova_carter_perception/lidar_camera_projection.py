import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, CameraInfo
from sensor_msgs_py import point_cloud2

from tf2_ros import Buffer, TransformListener, TransformException
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud


class LidarCameraProjection(Node):

    def __init__(self):
        super().__init__('lidar_camera_projection')

        self.lidar_topic = '/front_3d_lidar/lidar_points'
        self.camera_info_topic = '/front_stereo_camera/left/camera_info'

        self.camera_frame = 'front_stereo_camera_left_optical'

        # Camera calibration values
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.image_width = None
        self.image_height = None

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        # Camera calibration subscriber
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data
        )

        # LiDAR subscriber
        self.lidar_sub = self.create_subscription(
            PointCloud2,
            self.lidar_topic,
            self.lidar_callback,
            qos_profile_sensor_data
        )

        # Debug cloud visible in RViz
        self.fov_cloud_pub = self.create_publisher(
            PointCloud2,
            '/lidar/camera_fov_points',
            10
        )

        self.frame_count = 0

        self.get_logger().info(
            'LiDAR-camera projection node started'
        )


    def camera_info_callback(self, msg):

        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

        self.image_width = msg.width
        self.image_height = msg.height


    def lidar_callback(self, msg):

        # Cannot project until CameraInfo has arrived
        if self.fx is None:
            return

        try:
            # We want:
            #
            # LiDAR coordinates
            #       ↓
            # camera optical coordinates
            #
            transform = self.tf_buffer.lookup_transform(
                self.camera_frame,
                msg.header.frame_id,
                Time()
            )

        except TransformException as ex:
            self.get_logger().warning(
                f'TF unavailable: {ex}'
            )
            return

        # Transform the entire PointCloud2
        # into the camera optical frame.
        cloud_camera = do_transform_cloud(
            msg,
            transform
        )

        # Extract XYZ
        points = point_cloud2.read_points_numpy(
            cloud_camera,
            field_names=('x', 'y', 'z'),
            skip_nans=True
        )

        if len(points) == 0:
            return

        points = np.asarray(points)

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # Only points physically in front of the camera
        in_front = z > 0.05

        x = x[in_front]
        y = y[in_front]
        z = z[in_front]

        points_front = points[in_front]

        if len(points_front) == 0:
            return

        # 3D camera coordinates → 2D pixels
        u = self.fx * x / z + self.cx
        v = self.fy * y / z + self.cy

        # Keep points that actually fall inside the image
        inside_image = (
            (u >= 0) &
            (u < self.image_width) &
            (v >= 0) &
            (v < self.image_height)
        )

        fov_points = points_front[inside_image]

        # Publish only the points visible by the camera
        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = self.camera_frame

        output_cloud = point_cloud2.create_cloud_xyz32(
            header,
            fov_points[:, :3].tolist()
        )

        self.fov_cloud_pub.publish(output_cloud)

        self.frame_count += 1

        if self.frame_count % 10 == 0:
            self.get_logger().info(
                f'LiDAR points: {len(points)} | '
                f'In camera FOV: {len(fov_points)}'
            )


def main(args=None):

    rclpy.init(args=args)

    node = LidarCameraProjection()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()