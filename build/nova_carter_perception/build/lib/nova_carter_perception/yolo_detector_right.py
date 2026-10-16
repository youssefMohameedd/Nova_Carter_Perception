import os
import time

import cv2
import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

from ultralytics import YOLO


class RightYoloDetector(Node):

    def __init__(self):

        super().__init__('right_yolo_detector')

        # =====================================================
        # Configuration
        # =====================================================

        self.image_topic = (
            '/front_stereo_camera/right/image_raw'
        )

        self.detections_topic = (
            '/yolo/right/detections'
        )

        self.annotated_topic = (
            '/yolo/right/annotated_image'
        )

        self.model_path = os.path.expanduser(
            '~/projects/nova_carter_yolo/models/yolov8n.pt'
        )

        self.output_dir = os.path.expanduser(
            '~/projects/nova_carter_yolo/'
            'outputs/right_yolo'
        )

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        # Slightly stricter than the original 0.35
        # to reduce duplicate detections.
        self.confidence_threshold = 0.60

        self.iou_threshold = 0.45

        self.image_size = 640

        self.device = 'cpu'

        self.save_every_n_frames = 10

        # =====================================================
        # YOLO
        # =====================================================

        self.get_logger().info(
            f'Loading YOLO model: {self.model_path}'
        )

        self.model = YOLO(
            self.model_path
        )

        self.bridge = CvBridge()

        # =====================================================
        # ROS
        # =====================================================

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data
        )

        self.detection_pub = self.create_publisher(
            Detection2DArray,
            self.detections_topic,
            10
        )

        self.annotated_pub = self.create_publisher(
            Image,
            self.annotated_topic,
            10
        )

        self.frame_count = 0

        self.get_logger().info(
            'RIGHT-camera YOLO detector started'
        )

        self.get_logger().info(
            f'Input: {self.image_topic}'
        )

        self.get_logger().info(
            f'Detections: {self.detections_topic}'
        )

    # =========================================================
    # Image callback
    # =========================================================

    def image_callback(self, msg):

        try:

            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

        except Exception as ex:

            self.get_logger().error(
                f'cv_bridge error: {ex}'
            )

            return

        start_time = time.perf_counter()

        # =====================================================
        # YOLO inference
        # =====================================================

        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False
        )

        inference_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        if len(results) == 0:
            return

        result = results[0]

        # =====================================================
        # Detection2DArray
        # =====================================================

        detection_array = Detection2DArray()

        detection_array.header = msg.header

        parsed_detections = []

        # =====================================================
        # Convert YOLO boxes -> vision_msgs
        # =====================================================

        if result.boxes is not None:

            for box in result.boxes:

                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .tolist()
                )

                class_id = int(
                    box.cls[0].item()
                )

                confidence = float(
                    box.conf[0].item()
                )

                class_name = self.model.names[
                    class_id
                ]

                center_x = (
                    x1 + x2
                ) / 2.0

                center_y = (
                    y1 + y2
                ) / 2.0

                width = x2 - x1
                height = y2 - y1

                detection = Detection2D()

                detection.header = msg.header

                # Compatible with your current Jazzy
                # vision_msgs layout.
                detection.bbox.center.position.x = (
                    center_x
                )

                detection.bbox.center.position.y = (
                    center_y
                )

                detection.bbox.center.theta = 0.0

                detection.bbox.size_x = width
                detection.bbox.size_y = height

                hypothesis = (
                    ObjectHypothesisWithPose()
                )

                hypothesis.hypothesis.class_id = (
                    class_name
                )

                hypothesis.hypothesis.score = (
                    confidence
                )

                detection.results.append(
                    hypothesis
                )

                detection_array.detections.append(
                    detection
                )

                parsed_detections.append(
                    (
                        class_name,
                        confidence,
                        x1,
                        y1,
                        x2,
                        y2
                    )
                )

        # =====================================================
        # Publish standard ROS detections
        # =====================================================

        self.detection_pub.publish(
            detection_array
        )

        # =====================================================
        # Annotated YOLO image
        # =====================================================

        annotated_frame = result.plot()

        annotated_msg = (
            self.bridge.cv2_to_imgmsg(
                annotated_frame,
                encoding='bgr8'
            )
        )

        annotated_msg.header = msg.header

        self.annotated_pub.publish(
            annotated_msg
        )

        # =====================================================
        # Logging / exporting
        # =====================================================

        self.frame_count += 1

        if (
            self.frame_count
            % self.save_every_n_frames
            == 0
        ):

            output_path = os.path.join(
                self.output_dir,
                'latest_right_yolo.jpg'
            )

            cv2.imwrite(
                output_path,
                annotated_frame
            )

            self.get_logger().info(
                f'RIGHT Frame {self.frame_count} | '
                f'Detections={len(parsed_detections)} | '
                f'Inference={inference_ms:.1f} ms'
            )

            for (
                class_name,
                confidence,
                x1,
                y1,
                x2,
                y2
            ) in parsed_detections:

                self.get_logger().info(
                    f'  {class_name}: '
                    f'{confidence:.2f} | '
                    f'bbox='
                    f'[{x1:.1f}, '
                    f'{y1:.1f}, '
                    f'{x2:.1f}, '
                    f'{y2:.1f}]'
                )


def main(args=None):

    rclpy.init(args=args)

    node = RightYoloDetector()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()