import copy
import math

import rclpy
from rclpy.node import Node

from vision_msgs.msg import (
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)


class MultiCameraFusion(Node):

    def __init__(self):
        super().__init__('multi_camera_fusion')

        # =====================================================
        # Topics
        # =====================================================

        self.left_topic = (
            '/fusion/detections_3d'
        )

        self.right_topic = (
            '/fusion/right/detections_3d'
        )

        self.output_topic = (
            '/fusion/final/detections_3d'
        )

        # Both camera-LiDAR fusion nodes already publish
        # detections relative to base_link.
        self.output_frame = 'base_link'

        # =====================================================
        # Association settings
        # =====================================================

        # Maximum 3D separation for considering two
        # detections to represent the same physical object.
        #
        # Start generous because our current ROI-LiDAR
        # localization is still approximate.
        self.association_distance = 0.60

        # Do not combine left/right messages that are
        # excessively far apart in simulation time.
        self.max_time_difference = 0.50

        # =====================================================
        # Latest messages
        # =====================================================

        self.latest_left = None
        self.latest_right = None

        # Used to avoid publishing the exact same pair
        # repeatedly.
        self.last_fused_pair = None

        # =====================================================
        # ROS subscriptions
        # =====================================================

        self.left_sub = self.create_subscription(
            Detection3DArray,
            self.left_topic,
            self.left_callback,
            10
        )

        self.right_sub = self.create_subscription(
            Detection3DArray,
            self.right_topic,
            self.right_callback,
            10
        )

        # =====================================================
        # Publisher
        # =====================================================

        self.final_pub = self.create_publisher(
            Detection3DArray,
            self.output_topic,
            10
        )

        self.get_logger().info(
            'Multi-camera 3D fusion node started'
        )

        self.get_logger().info(
            f'LEFT:  {self.left_topic}'
        )

        self.get_logger().info(
            f'RIGHT: {self.right_topic}'
        )

        self.get_logger().info(
            f'FINAL: {self.output_topic}'
        )

        self.get_logger().info(
            f'Association distance = '
            f'{self.association_distance:.2f} m'
        )

    # =========================================================
    # Callbacks
    # =========================================================

    def left_callback(self, msg):

        self.latest_left = msg
        self.try_fusion()

    def right_callback(self, msg):

        self.latest_right = msg
        self.try_fusion()

    # =========================================================
    # Timestamp helpers
    # =========================================================

    def stamp_to_seconds(self, stamp):

        return (
            float(stamp.sec)
            + float(stamp.nanosec) * 1e-9
        )

    # =========================================================
    # Detection helpers
    # =========================================================

    def get_class_and_confidence(self, detection):

        if len(detection.results) == 0:

            return 'unknown', 0.0

        hypothesis = (
            detection.results[0].hypothesis
        )

        return (
            hypothesis.class_id,
            float(hypothesis.score)
        )

    def get_position(self, detection):

        # bbox.center is convenient because our previous
        # fusion nodes set it to the estimated object XYZ.
        return (
            float(detection.bbox.center.position.x),
            float(detection.bbox.center.position.y),
            float(detection.bbox.center.position.z)
        )

    def position_distance(
        self,
        detection_a,
        detection_b
    ):

        ax, ay, az = self.get_position(
            detection_a
        )

        bx, by, bz = self.get_position(
            detection_b
        )

        return math.sqrt(
            (ax - bx) ** 2
            + (ay - by) ** 2
            + (az - bz) ** 2
        )

    # =========================================================
    # Fuse one LEFT + one RIGHT detection
    # =========================================================

    def fuse_detection_pair(
        self,
        left_detection,
        right_detection,
        header
    ):

        left_class, left_conf = (
            self.get_class_and_confidence(
                left_detection
            )
        )

        right_class, right_conf = (
            self.get_class_and_confidence(
                right_detection
            )
        )

        lx, ly, lz = self.get_position(
            left_detection
        )

        rx, ry, rz = self.get_position(
            right_detection
        )

        # =====================================================
        # Confidence-weighted XYZ average
        # =====================================================

        total_weight = (
            left_conf + right_conf
        )

        if total_weight <= 1e-6:

            left_weight = 0.5
            right_weight = 0.5

        else:

            left_weight = (
                left_conf / total_weight
            )

            right_weight = (
                right_conf / total_weight
            )

        fused_x = (
            left_weight * lx
            + right_weight * rx
        )

        fused_y = (
            left_weight * ly
            + right_weight * ry
        )

        fused_z = (
            left_weight * lz
            + right_weight * rz
        )

        # For today's system we use the stronger
        # visual confidence rather than artificially
        # increasing it.
        fused_confidence = max(
            left_conf,
            right_conf
        )

        # =====================================================
        # Create standard Detection3D
        # =====================================================

        fused = Detection3D()

        fused.header = header

        hypothesis = ObjectHypothesisWithPose()

        hypothesis.hypothesis.class_id = (
            left_class
        )

        hypothesis.hypothesis.score = (
            fused_confidence
        )

        hypothesis.pose.pose.position.x = (
            fused_x
        )

        hypothesis.pose.pose.position.y = (
            fused_y
        )

        hypothesis.pose.pose.position.z = (
            fused_z
        )

        hypothesis.pose.pose.orientation.w = 1.0

        fused.results.append(
            hypothesis
        )

        # =====================================================
        # Bounding box center
        # =====================================================

        fused.bbox.center.position.x = (
            fused_x
        )

        fused.bbox.center.position.y = (
            fused_y
        )

        fused.bbox.center.position.z = (
            fused_z
        )

        fused.bbox.center.orientation.w = 1.0

        # =====================================================
        # Approximate fused 3D dimensions
        # =====================================================

        fused.bbox.size.x = (
            left_weight
            * float(left_detection.bbox.size.x)
            + right_weight
            * float(right_detection.bbox.size.x)
        )

        fused.bbox.size.y = (
            left_weight
            * float(left_detection.bbox.size.y)
            + right_weight
            * float(right_detection.bbox.size.y)
        )

        fused.bbox.size.z = (
            left_weight
            * float(left_detection.bbox.size.z)
            + right_weight
            * float(right_detection.bbox.size.z)
        )

        return fused

    # =========================================================
    # Main association
    # =========================================================

    def try_fusion(self):

        if self.latest_left is None:
            return

        if self.latest_right is None:
            return

        # =====================================================
        # Timestamp check
        # =====================================================

        left_time = self.stamp_to_seconds(
            self.latest_left.header.stamp
        )

        right_time = self.stamp_to_seconds(
            self.latest_right.header.stamp
        )

        time_difference = abs(
            left_time - right_time
        )

        if (
            time_difference
            > self.max_time_difference
        ):

            return

        current_pair = (
            left_time,
            right_time
        )

        if current_pair == self.last_fused_pair:
            return

        self.last_fused_pair = current_pair

        # =====================================================
        # Output message
        # =====================================================

        output = Detection3DArray()

        # Use timestamp of the newest observation.
        if right_time > left_time:

            output.header = copy.deepcopy(
                self.latest_right.header
            )

        else:

            output.header = copy.deepcopy(
                self.latest_left.header
            )

        output.header.frame_id = (
            self.output_frame
        )

        left_detections = list(
            self.latest_left.detections
        )

        right_detections = list(
            self.latest_right.detections
        )

        # =====================================================
        # Association candidates
        #
        # candidate =
        # (distance, left_index, right_index)
        # =====================================================

        candidates = []

        for left_index, left_detection in enumerate(
            left_detections
        ):

            left_class, _ = (
                self.get_class_and_confidence(
                    left_detection
                )
            )

            for right_index, right_detection in enumerate(
                right_detections
            ):

                right_class, _ = (
                    self.get_class_and_confidence(
                        right_detection
                    )
                )

                # Never associate different classes.
                if left_class != right_class:
                    continue

                distance = self.position_distance(
                    left_detection,
                    right_detection
                )

                if (
                    distance
                    <= self.association_distance
                ):

                    candidates.append(
                        (
                            distance,
                            left_index,
                            right_index
                        )
                    )

        # Closest matches get priority.
        candidates.sort(
            key=lambda item: item[0]
        )

        matched_left = set()
        matched_right = set()

        # =====================================================
        # Fuse matched pairs
        # =====================================================

        for (
            distance,
            left_index,
            right_index
        ) in candidates:

            if left_index in matched_left:
                continue

            if right_index in matched_right:
                continue

            left_detection = (
                left_detections[left_index]
            )

            right_detection = (
                right_detections[right_index]
            )

            fused_detection = (
                self.fuse_detection_pair(
                    left_detection,
                    right_detection,
                    output.header
                )
            )

            output.detections.append(
                fused_detection
            )

            matched_left.add(
                left_index
            )

            matched_right.add(
                right_index
            )

            class_name, _ = (
                self.get_class_and_confidence(
                    fused_detection
                )
            )

            x, y, z = self.get_position(
                fused_detection
            )

            range_value = math.sqrt(
                x*x + y*y + z*z
            )

            self.get_logger().info(
                f'MATCHED {class_name} | '
                f'L#{left_index + 1} + '
                f'R#{right_index + 1} | '
                f'dXYZ={distance:.2f} m | '
                f'XYZ=[{x:.2f}, '
                f'{y:.2f}, '
                f'{z:.2f}] | '
                f'Range={range_value:.2f} m'
            )

        # =====================================================
        # Keep objects seen ONLY by LEFT camera
        # =====================================================

        for index, detection in enumerate(
            left_detections
        ):

            if index in matched_left:
                continue

            single_detection = copy.deepcopy(
                detection
            )

            single_detection.header = (
                output.header
            )

            output.detections.append(
                single_detection
            )

            class_name, confidence = (
                self.get_class_and_confidence(
                    single_detection
                )
            )

            self.get_logger().info(
                f'LEFT ONLY | '
                f'{class_name} '
                f'({confidence:.2f})'
            )

        # =====================================================
        # Keep objects seen ONLY by RIGHT camera
        # =====================================================

        for index, detection in enumerate(
            right_detections
        ):

            if index in matched_right:
                continue

            single_detection = copy.deepcopy(
                detection
            )

            single_detection.header = (
                output.header
            )

            output.detections.append(
                single_detection
            )

            class_name, confidence = (
                self.get_class_and_confidence(
                    single_detection
                )
            )

            self.get_logger().info(
                f'RIGHT ONLY | '
                f'{class_name} '
                f'({confidence:.2f})'
            )

        # =====================================================
        # Publish
        # =====================================================

        self.final_pub.publish(
            output
        )

        self.get_logger().info(
            'FINAL FUSION | '
            f'Left={len(left_detections)} | '
            f'Right={len(right_detections)} | '
            f'Matched={len(matched_left)} | '
            f'Final={len(output.detections)}'
        )


def main(args=None):

    rclpy.init(args=args)

    node = MultiCameraFusion()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()