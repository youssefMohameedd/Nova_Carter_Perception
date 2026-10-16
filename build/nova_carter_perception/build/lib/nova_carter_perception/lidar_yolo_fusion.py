import os

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from sensor_msgs_py import point_cloud2

from vision_msgs.msg import (
    Detection2DArray,
    Detection3DArray,
    Detection3D,
    ObjectHypothesisWithPose,
)

from cv_bridge import CvBridge

from tf2_ros import (
    Buffer,
    TransformListener,
    TransformException,
)


class LidarYoloFusion(Node):

    def __init__(self):
        super().__init__('lidar_yolo_fusion')

        # =====================================================
        # Topics / frames
        # =====================================================

        self.lidar_topic = (
            '/front_3d_lidar/lidar_points'
        )

        self.camera_info_topic = (
            '/front_stereo_camera/left/camera_info'
        )

        self.image_topic = (
            '/front_stereo_camera/left/image_raw'
        )

        self.detections_topic = (
            '/yolo/detections'
        )

        self.camera_frame = (
            'front_stereo_camera_left_optical'
        )

        # Final fused objects will be expressed
        # relative to the robot.
        self.output_frame = 'base_link'

        # =====================================================
        # Output directory
        # =====================================================

        self.output_dir = os.path.expanduser(
            '~/projects/nova_carter_yolo/'
            'outputs/fusion_debug'
        )

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        # Save debug image every N fusion cycles.
        # Use 1 during final testing if you want every frame.
        self.save_every_n_frames = 10

        # Draw every Nth visible LiDAR point
        # so the RGB image is still readable.
        self.draw_stride = 2

        # =====================================================
        # Camera intrinsics
        # =====================================================

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.image_width = None
        self.image_height = None

        # =====================================================
        # Latest incoming data
        # =====================================================

        self.latest_image_msg = None
        self.latest_detections = None

        self.bridge = CvBridge()

        # =====================================================
        # Input-status flags
        # =====================================================

        self.camera_info_received = False
        self.image_received = False
        self.detections_received = False
        self.lidar_received = False

        # =====================================================
        # TF
        # =====================================================

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        # =====================================================
        # Subscribers
        # =====================================================

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data
        )

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            self.detections_topic,
            self.detection_callback,
            10
        )

        self.lidar_sub = self.create_subscription(
            PointCloud2,
            self.lidar_topic,
            self.lidar_callback,
            qos_profile_sensor_data
        )

        # =====================================================
        # Publishers
        # =====================================================

        # LiDAR points geometrically inside camera FOV
        self.fov_points_pub = self.create_publisher(
            PointCloud2,
            '/fusion/camera_fov_points',
            10
        )

        # Combined LiDAR points belonging to YOLO ROIs
        self.object_points_pub = self.create_publisher(
            PointCloud2,
            '/fusion/object_points',
            10
        )

        # Annotated camera image
        self.debug_image_pub = self.create_publisher(
            Image,
            '/fusion/debug_image',
            10
        )

        # STANDARD multi-object 3D output
        self.detections_3d_pub = self.create_publisher(
            Detection3DArray,
            '/fusion/detections_3d',
            10
        )

        # =====================================================
        # Visualization
        # OpenCV uses BGR
        # =====================================================

        self.object_colors = [
            (0, 0, 255),       # red
            (255, 0, 255),     # magenta
            (0, 165, 255),     # orange
            (255, 255, 0),     # cyan
            (255, 0, 0),       # blue
            (0, 255, 255),     # yellow
        ]

        self.fusion_frame_count = 0

        # Only prints while something is missing.
        self.status_timer = self.create_timer(
            3.0,
            self.print_input_status
        )

        self.get_logger().info(
            'LiDAR-YOLO fusion node started'
        )

        self.get_logger().info(
            f'Debug images: {self.output_dir}'
        )

        self.get_logger().info(
            '3D detections topic: '
            '/fusion/detections_3d'
        )

    # =========================================================
    # Input status
    # =========================================================

    def print_input_status(self):

        all_ready = (
            self.camera_info_received
            and self.image_received
            and self.detections_received
            and self.lidar_received
        )

        if not all_ready:

            self.get_logger().info(
                'Waiting for inputs | '
                f'CameraInfo={self.camera_info_received} | '
                f'RGB={self.image_received} | '
                f'YOLO={self.detections_received} | '
                f'LiDAR={self.lidar_received}'
            )

    # =========================================================
    # CameraInfo callback
    # =========================================================

    def camera_info_callback(self, msg):

        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])

        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

        self.image_width = int(msg.width)
        self.image_height = int(msg.height)

        self.camera_info_received = True

    # =========================================================
    # RGB image callback
    # =========================================================

    def image_callback(self, msg):

        self.latest_image_msg = msg
        self.image_received = True

    # =========================================================
    # YOLO Detection2DArray callback
    # =========================================================

    def detection_callback(self, msg):

        self.latest_detections = msg
        self.detections_received = True

    # =========================================================
    # BoundingBox2D center helper
    # =========================================================

    def get_bbox_center(self, bbox):

        center = bbox.center

        # Compatibility with vision_msgs versions
        # using Pose2D-style center.position.
        if hasattr(center, 'position'):

            return (
                float(center.position.x),
                float(center.position.y)
            )

        return (
            float(center.x),
            float(center.y)
        )

    # =========================================================
    # Quaternion -> rotation matrix
    # =========================================================

    def quaternion_to_rotation_matrix(self, q):

        x = float(q.x)
        y = float(q.y)
        z = float(q.z)
        w = float(q.w)

        return np.array(
            [
                [
                    1.0 - 2.0 * (y*y + z*z),
                    2.0 * (x*y - z*w),
                    2.0 * (x*z + y*w)
                ],
                [
                    2.0 * (x*y + z*w),
                    1.0 - 2.0 * (x*x + z*z),
                    2.0 * (y*z - x*w)
                ],
                [
                    2.0 * (x*z - y*w),
                    2.0 * (y*z + x*w),
                    1.0 - 2.0 * (x*x + y*y)
                ]
            ],
            dtype=np.float64
        )

    # =========================================================
    # Transform Nx3 points using TransformStamped
    # =========================================================

    def transform_points(
        self,
        points,
        transform
    ):

        q = transform.transform.rotation
        translation = transform.transform.translation

        R = self.quaternion_to_rotation_matrix(q)

        t = np.array(
            [
                translation.x,
                translation.y,
                translation.z
            ],
            dtype=np.float64
        )

        return (
            R @ points.T
        ).T + t

    # =========================================================
    # PointCloud2 publisher helper
    # =========================================================

    def publish_xyz_cloud(
        self,
        publisher,
        points,
        source_msg,
        frame_id
    ):

        header = Header()

        header.stamp = source_msg.header.stamp
        header.frame_id = frame_id

        points = np.asarray(
            points,
            dtype=np.float32
        ).reshape(-1, 3)

        cloud_msg = point_cloud2.create_cloud_xyz32(
            header,
            points.tolist()
        )

        publisher.publish(cloud_msg)

    # =========================================================
    # Main LiDAR callback
    # =========================================================

    def lidar_callback(self, msg):

        self.lidar_received = True

        # -----------------------------------------------------
        # Wait for all required inputs
        # -----------------------------------------------------

        if self.fx is None:
            return

        if self.latest_detections is None:
            return

        if self.latest_image_msg is None:
            return

        # -----------------------------------------------------
        # 1. TF transforms
        # -----------------------------------------------------

        try:

            # LiDAR coordinates -> camera optical coordinates
            lidar_to_camera = self.tf_buffer.lookup_transform(
                self.camera_frame,
                msg.header.frame_id,
                Time()
            )

            # Camera optical coordinates -> base_link
            camera_to_base = self.tf_buffer.lookup_transform(
                self.output_frame,
                self.camera_frame,
                Time()
            )

        except TransformException as ex:

            self.get_logger().warning(
                f'TF unavailable: {ex}'
            )

            return

        # -----------------------------------------------------
        # 2. Read LiDAR XYZ
        # -----------------------------------------------------

        try:

            points_lidar = (
                point_cloud2.read_points_numpy(
                    msg,
                    field_names=('x', 'y', 'z'),
                    skip_nans=True
                )
            )

        except Exception as ex:

            self.get_logger().error(
                f'Could not read PointCloud2: {ex}'
            )

            return

        points_lidar = np.asarray(
            points_lidar,
            dtype=np.float64
        )

        if points_lidar.size == 0:
            return

        points_lidar = points_lidar.reshape(
            -1,
            3
        )

        # -----------------------------------------------------
        # 3. LiDAR XYZ -> Camera XYZ
        # -----------------------------------------------------

        points_camera = self.transform_points(
            points_lidar,
            lidar_to_camera
        )

        X = points_camera[:, 0]
        Y = points_camera[:, 1]
        Z = points_camera[:, 2]

        # -----------------------------------------------------
        # 4. Remove points behind the camera
        # -----------------------------------------------------

        in_front = Z > 0.05

        points_camera = points_camera[in_front]

        X = X[in_front]
        Y = Y[in_front]
        Z = Z[in_front]

        if len(points_camera) == 0:
            return

        # -----------------------------------------------------
        # 5. Project camera XYZ -> pixel coordinates
        # -----------------------------------------------------

        u = (
            self.fx * X / Z
            + self.cx
        )

        v = (
            self.fy * Y / Z
            + self.cy
        )

        # -----------------------------------------------------
        # 6. Keep only points inside image FOV
        # -----------------------------------------------------

        inside_image = (
            (u >= 0.0) &
            (u < self.image_width) &
            (v >= 0.0) &
            (v < self.image_height)
        )

        visible_points = points_camera[
            inside_image
        ]

        visible_u = u[
            inside_image
        ]

        visible_v = v[
            inside_image
        ]

        if len(visible_points) == 0:
            return

        # Publish camera-FOV cloud
        self.publish_xyz_cloud(
            self.fov_points_pub,
            visible_points,
            msg,
            self.camera_frame
        )

        # -----------------------------------------------------
        # 7. Get current RGB image
        # -----------------------------------------------------

        try:

            frame = self.bridge.imgmsg_to_cv2(
                self.latest_image_msg,
                desired_encoding='bgr8'
            ).copy()

        except Exception as ex:

            self.get_logger().error(
                f'cv_bridge failed: {ex}'
            )

            return

        # -----------------------------------------------------
        # 8. Draw all camera-visible LiDAR points GREEN
        # -----------------------------------------------------

        draw_u = visible_u[
            ::self.draw_stride
        ].astype(np.int32)

        draw_v = visible_v[
            ::self.draw_stride
        ].astype(np.int32)

        for px, py in zip(
            draw_u,
            draw_v
        ):

            cv2.circle(
                frame,
                (int(px), int(py)),
                1,
                (0, 255, 0),
                -1
            )

        # -----------------------------------------------------
        # 9. Create standard Detection3DArray
        # -----------------------------------------------------

        detections_3d_msg = Detection3DArray()

        detections_3d_msg.header.stamp = (
            msg.header.stamp
        )

        detections_3d_msg.header.frame_id = (
            self.output_frame
        )

        all_object_points = []

        # -----------------------------------------------------
        # 10. Process every YOLO detection independently
        # -----------------------------------------------------

        for detection_index, detection in enumerate(
            self.latest_detections.detections
        ):

            object_number = detection_index + 1

            object_color = self.object_colors[
                detection_index
                % len(self.object_colors)
            ]

            bbox = detection.bbox

            center_x, center_y = (
                self.get_bbox_center(bbox)
            )

            width = float(bbox.size_x)
            height = float(bbox.size_y)

            xmin = center_x - width / 2.0
            xmax = center_x + width / 2.0

            ymin = center_y - height / 2.0
            ymax = center_y + height / 2.0

            # Drawing coordinates
            xmin_draw = max(
                0,
                int(round(xmin))
            )

            ymin_draw = max(
                0,
                int(round(ymin))
            )

            xmax_draw = min(
                frame.shape[1] - 1,
                int(round(xmax))
            )

            ymax_draw = min(
                frame.shape[0] - 1,
                int(round(ymax))
            )

            # -------------------------------------------------
            # YOLO class / confidence
            # -------------------------------------------------

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

            # -------------------------------------------------
            # Draw YOLO box
            # -------------------------------------------------

            cv2.rectangle(
                frame,
                (xmin_draw, ymin_draw),
                (xmax_draw, ymax_draw),
                object_color,
                2
            )

            # -------------------------------------------------
            # 11. LiDAR points whose projection lies in ROI
            # -------------------------------------------------

            inside_bbox = (
                (visible_u >= xmin) &
                (visible_u <= xmax) &
                (visible_v >= ymin) &
                (visible_v <= ymax)
            )

            object_points_camera = visible_points[
                inside_bbox
            ]

            object_u = visible_u[
                inside_bbox
            ]

            object_v = visible_v[
                inside_bbox
            ]

            # -------------------------------------------------
            # Draw associated points using object color
            # -------------------------------------------------

            for px, py in zip(
                object_u.astype(np.int32),
                object_v.astype(np.int32)
            ):

                cv2.circle(
                    frame,
                    (int(px), int(py)),
                    2,
                    object_color,
                    -1
                )

            # -------------------------------------------------
            # No LiDAR for this detection
            # -------------------------------------------------

            if len(object_points_camera) == 0:

                label = (
                    f'{class_name} #{object_number} '
                    f'{confidence:.2f} | no LiDAR'
                )

                cv2.putText(
                    frame,
                    label,
                    (
                        xmin_draw,
                        max(20, ymin_draw - 8)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    object_color,
                    1,
                    cv2.LINE_AA
                )

                continue

            all_object_points.append(
                object_points_camera
            )

            # -------------------------------------------------
            # 12. Camera-frame object estimate
            # -------------------------------------------------

            depths = object_points_camera[:, 2]

            median_camera = np.median(
                object_points_camera,
                axis=0
            )

            x_cam = float(
                median_camera[0]
            )

            y_cam = float(
                median_camera[1]
            )

            z_cam = float(
                median_camera[2]
            )

            # Straight-line range from camera
            range_camera = float(
                np.linalg.norm(
                    median_camera
                )
            )

            z_min = float(
                np.min(depths)
            )

            z_median = float(
                np.median(depths)
            )

            z_max = float(
                np.max(depths)
            )

            # -------------------------------------------------
            # 13. Camera points -> base_link
            # -------------------------------------------------

            object_points_base = self.transform_points(
                object_points_camera,
                camera_to_base
            )

            median_base = np.median(
                object_points_base,
                axis=0
            )

            x_base = float(
                median_base[0]
            )

            y_base = float(
                median_base[1]
            )

            z_base = float(
                median_base[2]
            )

            range_base = float(
                np.linalg.norm(
                    median_base
                )
            )

            # -------------------------------------------------
            # 14. Create standard Detection3D
            # -------------------------------------------------

            detection_3d = Detection3D()

            detection_3d.header.stamp = (
                msg.header.stamp
            )

            detection_3d.header.frame_id = (
                self.output_frame
            )

            # IMPORTANT:
            # Do not set Detection3D.id yet.
            #
            # object_number is only a frame-local visualization
            # number, NOT a persistent tracking ID.

            # Classification + 3D pose
            hypothesis_3d = ObjectHypothesisWithPose()

            hypothesis_3d.hypothesis.class_id = (
                class_name
            )

            hypothesis_3d.hypothesis.score = (
                confidence
            )

            hypothesis_3d.pose.pose.position.x = (
                x_base
            )

            hypothesis_3d.pose.pose.position.y = (
                y_base
            )

            hypothesis_3d.pose.pose.position.z = (
                z_base
            )

            # Object orientation is not estimated yet.
            # Identity quaternion is used.
            hypothesis_3d.pose.pose.orientation.x = 0.0
            hypothesis_3d.pose.pose.orientation.y = 0.0
            hypothesis_3d.pose.pose.orientation.z = 0.0
            hypothesis_3d.pose.pose.orientation.w = 1.0

            detection_3d.results.append(
                hypothesis_3d
            )

            # -------------------------------------------------
            # Detection3D bounding-box center
            # -------------------------------------------------

            detection_3d.bbox.center.position.x = (
                x_base
            )

            detection_3d.bbox.center.position.y = (
                y_base
            )

            detection_3d.bbox.center.position.z = (
                z_base
            )

            detection_3d.bbox.center.orientation.x = 0.0
            detection_3d.bbox.center.orientation.y = 0.0
            detection_3d.bbox.center.orientation.z = 0.0
            detection_3d.bbox.center.orientation.w = 1.0

            # -------------------------------------------------
            # Approximate BoundingBox3D dimensions
            #
            # Percentiles are safer than raw min/max because
            # a few background points may exist in the ROI.
            # -------------------------------------------------

            lower = np.percentile(
                object_points_base,
                10,
                axis=0
            )

            upper = np.percentile(
                object_points_base,
                90,
                axis=0
            )

            bbox_size = upper - lower

            detection_3d.bbox.size.x = max(
                float(bbox_size[0]),
                0.05
            )

            detection_3d.bbox.size.y = max(
                float(bbox_size[1]),
                0.05
            )

            detection_3d.bbox.size.z = max(
                float(bbox_size[2]),
                0.05
            )

            detections_3d_msg.detections.append(
                detection_3d
            )

            # -------------------------------------------------
            # 15. Draw estimated camera-frame median YELLOW
            # -------------------------------------------------

            if z_cam > 0.05:

                median_u = int(
                    round(
                        self.fx
                        * x_cam
                        / z_cam
                        + self.cx
                    )
                )

                median_v = int(
                    round(
                        self.fy
                        * y_cam
                        / z_cam
                        + self.cy
                    )
                )

                if (
                    0 <= median_u < frame.shape[1]
                    and
                    0 <= median_v < frame.shape[0]
                ):

                    cv2.circle(
                        frame,
                        (median_u, median_v),
                        6,
                        (0, 255, 255),
                        2
                    )

                    cv2.drawMarker(
                        frame,
                        (median_u, median_v),
                        (0, 255, 255),
                        markerType=cv2.MARKER_CROSS,
                        markerSize=12,
                        thickness=2
                    )

            # -------------------------------------------------
            # 16. Debug image labels
            # -------------------------------------------------

            label_1 = (
                f'{class_name} #{object_number} '
                f'{confidence:.2f}'
            )

            label_2 = (
                f'RangeCam={range_camera:.2f}m '
                f'N={len(object_points_camera)}'
            )

            label_3 = (
                f'XYZbase='
                f'({x_base:.2f},'
                f'{y_base:.2f},'
                f'{z_base:.2f})'
            )

            # Put text inside the box near its upper edge.
            text_y_1 = max(
                20,
                ymin_draw + 18
            )

            text_y_2 = min(
                frame.shape[0] - 25,
                text_y_1 + 18
            )

            text_y_3 = min(
                frame.shape[0] - 5,
                text_y_2 + 18
            )

            cv2.putText(
                frame,
                label_1,
                (xmin_draw + 3, text_y_1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                object_color,
                1,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                label_2,
                (xmin_draw + 3, text_y_2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                object_color,
                1,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                label_3,
                (xmin_draw + 3, text_y_3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                object_color,
                1,
                cv2.LINE_AA
            )

            # -------------------------------------------------
            # 17. Terminal output
            # -------------------------------------------------

            self.get_logger().info(
                f'{class_name} #{object_number} '
                f'({confidence:.2f}) | '
                f'Points={len(object_points_camera)} | '
                f'Z[min/med/max]='
                f'{z_min:.2f}/'
                f'{z_median:.2f}/'
                f'{z_max:.2f} m | '
                f'RangeCam={range_camera:.2f} m | '
                f'XYZbase='
                f'[{x_base:.2f}, '
                f'{y_base:.2f}, '
                f'{z_base:.2f}] m | '
                f'RangeBase={range_base:.2f} m'
            )

        # =====================================================
        # 18. Publish Detection3DArray
        # =====================================================

        self.detections_3d_pub.publish(
            detections_3d_msg
        )

        # =====================================================
        # 19. Publish combined ROI point cloud
        # =====================================================

        if len(all_object_points) > 0:

            object_cloud = np.vstack(
                all_object_points
            )

        else:

            object_cloud = np.empty(
                (0, 3),
                dtype=np.float32
            )

        self.publish_xyz_cloud(
            self.object_points_pub,
            object_cloud,
            msg,
            self.camera_frame
        )

        # =====================================================
        # 20. Legend
        # =====================================================

        cv2.rectangle(
            frame,
            (0, 0),
            (285, 68),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            'GREEN: all camera-FOV LiDAR',
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            'COLORED: YOLO ROI + associated LiDAR',
            (8, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            'YELLOW CROSS: median 3D estimate',
            (8, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

        # =====================================================
        # 21. Publish debug image
        # =====================================================

        debug_msg = self.bridge.cv2_to_imgmsg(
            frame,
            encoding='bgr8'
        )

        debug_msg.header = (
            self.latest_image_msg.header
        )

        self.debug_image_pub.publish(
            debug_msg
        )

        # =====================================================
        # 22. Export debug image
        # =====================================================

        self.fusion_frame_count += 1

        if (
            self.fusion_frame_count
            % self.save_every_n_frames
            == 0
        ):

            latest_path = os.path.join(
                self.output_dir,
                'latest_fusion.jpg'
            )

            numbered_path = os.path.join(
                self.output_dir,
                (
                    f'fusion_'
                    f'{self.fusion_frame_count:06d}.jpg'
                )
            )

            latest_success = cv2.imwrite(
                latest_path,
                frame
            )

            numbered_success = cv2.imwrite(
                numbered_path,
                frame
            )

            if (
                latest_success
                and numbered_success
            ):

                self.get_logger().info(
                    f'Saved debug frame: '
                    f'{numbered_path}'
                )

            else:

                self.get_logger().error(
                    'Failed to export fusion image'
                )


def main(args=None):

    rclpy.init(args=args)

    node = LidarYoloFusion()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()