import os
import time

import cv2
import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ultralytics import YOLO

from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)


class YoloDetector(Node):

    def __init__(self):
        super().__init__('yolo_detector')

        # -------------------------------------------------
        # 1. Configuration
        # -------------------------------------------------

        self.image_topic = (
            '/front_stereo_camera/left/image_raw'
        )

        self.model_path = os.path.expanduser(
            '~/projects/nova_carter_yolo/models/yolov8n.pt'
        )

        self.output_dir = os.path.expanduser(
            '~/projects/nova_carter_yolo/outputs/live_yolo'
        )

        self.output_image = os.path.join(
            self.output_dir,
            'latest_annotated.jpg'
        )

        self.device = 'cpu'
        self.confidence_threshold = 0.60
        self.image_size = 640
        self.confidence_threshold = 0.50
        self.iou_threshold = 0.45

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        # -------------------------------------------------
        # 2. ROS / OpenCV / YOLO initialization
        # -------------------------------------------------

        self.bridge = CvBridge()

        self.model = YOLO(
            self.model_path
        )

        self.frame_count = 0

        # -------------------------------------------------
        # 3. ROS subscriber
        # -------------------------------------------------

        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data
        )

        # -------------------------------------------------
        # 4. ROS publishers
        # -------------------------------------------------

        # Structured semantic detections:
        # reliable queue depth = 10 // we may use best effort as well, but this is a low-bandwidth topic, so unifying it
        # with the annotated image topic is structurally-readable wise.
        self.detection_publisher = self.create_publisher(
        Detection2DArray,
        '/yolo/detections',
        10
        )
        # High-bandwidth visualization stream:
        # reliable queue depth = 10   // rtransmits lost data instead of dropping it, which is important for annotated images
        self.annotated_publisher = self.create_publisher(
            Image,
            '/yolo/annotated_image',
            10
        )

        # -------------------------------------------------
        # 5. Startup information
        # -------------------------------------------------

        self.get_logger().info(
            f'Listening to {self.image_topic}'
        )

        self.get_logger().info(
            f'YOLO model: {self.model_path}'
        )

        self.get_logger().info(
            f'YOLO device: {self.device}'
        )


    def image_callback(self, msg):

        self.frame_count += 1

        # =================================================
        # 1. Convert ROS image -> OpenCV image
        # =================================================

        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

        except Exception as error:

            self.get_logger().error(
                f'cv_bridge conversion failed: {error}'
            )

            return

        # =================================================
        # 2. Run YOLO inference
        # =================================================

        start_time = time.perf_counter()

        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False
        )

        inference_time = (
            time.perf_counter() - start_time
        )

        result = results[0]

        # =================================================
        # 3. Create Detection2DArray for this frame
        # =================================================

        detection_array = Detection2DArray()

        # Preserve original camera timestamp + frame_id.
        detection_array.header = msg.header

        # Useful for terminal logging later.
        parsed_detections = []

        # =================================================
        # 4. Convert every YOLO detection -> ROS detection
        # =================================================

        for box in result.boxes:

            # YOLO:
            # [x1, y1, x2, y2]
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

            # Convert YOLO corner representation
            # into ROS center + dimensions.
            center_x = (
                x1 + x2
            ) / 2.0

            center_y = (
                y1 + y2
            ) / 2.0

            width = (
                x2 - x1
            )

            height = (
                y2 - y1
            )

            # ---------------------------------------------
            # Create Detection2D
            # ---------------------------------------------

            detection = Detection2D()

            detection.header = msg.header

            detection.bbox.center.position.x = float(
                center_x
            )

            detection.bbox.center.position.y = float(
                center_y
            )

            # YOLO boxes are axis aligned,
            # therefore no rotation.
            detection.bbox.center.theta = 0.0

            detection.bbox.size_x = float(
                width
            )

            detection.bbox.size_y = float(
                height
            )

            # ---------------------------------------------
            # Add class + confidence hypothesis
            # ---------------------------------------------

            hypothesis = ObjectHypothesisWithPose()

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

            # Store once so we don't parse the box again
            # during logging.
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

        # =================================================
        # 5. Publish structured YOLO detections
        # =================================================

        self.detection_publisher.publish(
            detection_array
        )

        # ==============================================ee===
        # 6. Generate annotated image
        # =================================================

        annotated_frame = result.plot() ## this burns out the bounding box into the annotated_image 

        # =================================================
        # 7. Convert OpenCV image -> ROS Image
        # =================================================

        annotated_msg = (
            self.bridge.cv2_to_imgmsg(
                annotated_frame,
                encoding='bgr8'
            )
        )

        annotated_msg.header = msg.header

        # =================================================
        # 8. Publish annotated image
        # =================================================

        self.annotated_publisher.publish(
            annotated_msg
        )

        # =================================================
        # 9. Periodic save + logging
        # =================================================

        if self.frame_count % 10 == 0:

            cv2.imwrite(
                self.output_image,
                annotated_frame
            )

            self.get_logger().info(
                f'Frame {self.frame_count} | '
                f'Detections: '
                f'{len(parsed_detections)} | '
                f'Inference: '
                f'{inference_time * 1000:.1f} ms'
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

    rclpy.init(
        args=args
    )

    node = YoloDetector()

    try:
        rclpy.spin(
            node
        )

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        rclpy.try_shutdown()


if __name__ == '__main__':
    main()