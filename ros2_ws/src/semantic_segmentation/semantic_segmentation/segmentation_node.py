import rclpy
import cv2
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

from rgbd_mapping.semantics.inference import SemanticSegmenter

class SegmentationNode(Node):

    def __init__(self):
        super().__init__("semantic_segmentation_node")

        self.bridge = CvBridge()

        self.rgb_subscription = self.create_subscription(
            Image,
            "/rgbd_replay/rgb",
            self.rgb_callback,
            10,
        )

        self.labels_publisher = self.create_publisher(
            Image,
            "/semantic_segmentation/labels",
            10,
        )

        self.confidence_publisher = self.create_publisher(
            Image,
            "/semantic_segmentation/confidence",
            10,
        )

        self.get_logger().info("Loading semantic segmentation model...")

        self.segmenter = SemanticSegmenter(
            device="cpu"
        )

        self.get_logger().info("Semantic segmentation model ready.")


    def rgb_callback(self, msg):
        image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding = "bgr8")

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        prediction = self.segmenter.predict(
            image_rgb
        )

        labels = prediction.labels.astype(
            np.uint8,
            copy=False,
        )

        confidence = prediction.confidence.astype(
            np.float32,
            copy=False,
        )

        labels_msg = self.bridge.cv2_to_imgmsg(
            labels,
            encoding="8UC1",
        )

        confidence_msg = self.bridge.cv2_to_imgmsg(
            confidence,
            encoding="32FC1",
        )

        labels_msg.header.stamp = msg.header.stamp
        labels_msg.header.frame_id = msg.header.frame_id

        confidence_msg.header.stamp = msg.header.stamp
        confidence_msg.header.frame_id = msg.header.frame_id

        self.labels_publisher.publish(labels_msg)
        self.confidence_publisher.publish(confidence_msg)

        self.get_logger().info(
            f"RGB={image_rgb.shape}, "
            f"labels={prediction.labels.shape}, "
            f"confidence={prediction.confidence.shape}, "
            f"label_dtype={prediction.labels.dtype}, "
            f"confidence_dtype={prediction.confidence.dtype}"
        )

        self.get_logger().info(
            f"Published segmentation: "
            f"labels={labels.shape}/{labels.dtype}, "
            f"confidence={confidence.shape}/{confidence.dtype}, "
            f"stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec}"
        )


def main(args=None):
    rclpy.init(args=args)

    node = SegmentationNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()