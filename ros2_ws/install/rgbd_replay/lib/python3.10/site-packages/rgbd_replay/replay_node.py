import rclpy
from rclpy.node import Node
from pathlib import Path

from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped

import cv2
from cv_bridge import CvBridge
import csv


def load_associated_records(
    csv_path: Path,
    dataset_root: Path,
):
    records = []

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            record = {
                "frame_index": int(row["frame_index"]),

                "rgb_timestamp": float(row["rgb_timestamp"]),
                "rgb_path": dataset_root / row["rgb_path"],

                "depth_timestamp": float(row["depth_timestamp"]),
                "depth_path": dataset_root / row["depth_path"],

                "pose_timestamp": float(row["pose_timestamp"]),

                "tx": float(row["tx"]),
                "ty": float(row["ty"]),
                "tz": float(row["tz"]),

                "qx": float(row["qx"]),
                "qy": float(row["qy"]),
                "qz": float(row["qz"]),
                "qw": float(row["qw"]),
            }

            records.append(record)

    return records

class RGBDReplayNode(Node):

    def __init__(self):
        super().__init__("rgbd_replay_node")

        self.frame_publisher = self.create_publisher(
            Int32,
            "/rgbd_replay/frame_index",
            10,
        )

        self.rgb_publisher = self.create_publisher(
            Image,
            "/rgbd_replay/rgb",
            10,
        )

        self.depth_publisher = self.create_publisher(
            Image,
            "/rgbd_replay/depth",
            10,
        )

        self.pose_publisher = self.create_publisher(
            PoseStamped,
            "/rgbd_replay/pose",
            10,
        )

        self.dataset_root = Path("/home/yutao/Desktop/my_proj/rgbd-semantic-mapping/data/raw/rgbd_dataset_freiburg1_xyz")
        self.manifest_root = Path("/home/yutao/Desktop/my_proj/rgbd-semantic-mapping/data/processed/tum_associated_records.csv")
        self.records = load_associated_records(
            self.manifest_root,
            self.dataset_root,
        )

        self.current_record_index = 0

        self.get_logger().info(
            f"Loaded {len(self.records)} associated RGB-D-pose records."
        )

        self.bridge = CvBridge()

        self.timer = self.create_timer(
            0.5,
            self.timer_callback,
        )

    def timer_callback(self):
        if self.current_record_index >= len(self.records):
            self.get_logger().info(
                "Replay finished."
            )
            self.timer.cancel()
            return
    
        index = self.current_record_index
        record = self.records[index]

        # publish frame index
        index_msg = Int32()
        index_msg.data = record["frame_index"]
        self.frame_publisher.publish(index_msg)

        # load image
        image = cv2.imread(str(record["rgb_path"]), cv2.IMREAD_COLOR,)
        depth = cv2.imread(str(record["depth_path"]), cv2.IMREAD_UNCHANGED,)

        # convert to ROS Image
        image_msg = self.bridge.cv2_to_imgmsg(
            image,
            encoding="bgr8",
        )

        depth_msg = self.bridge.cv2_to_imgmsg(
            depth,
            encoding="16UC1",
        )

        stamp = self.get_clock().now().to_msg()
        image_msg.header.stamp = stamp
        image_msg.header.frame_id = "camera_rgb_optical_frame"

        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = "camera_depth_frame"

        pose_msg = PoseStamped()

        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = "world"

        pose_msg.pose.position.x = record["tx"]
        pose_msg.pose.position.y = record["ty"]
        pose_msg.pose.position.z = record["tz"]

        pose_msg.pose.orientation.x = record["qx"]
        pose_msg.pose.orientation.y = record["qy"]
        pose_msg.pose.orientation.z = record["qz"]
        pose_msg.pose.orientation.w = record["qw"]

        self.rgb_publisher.publish(image_msg)
        self.depth_publisher.publish(depth_msg)
        self.pose_publisher.publish(pose_msg)

        self.current_record_index += 1


def main(args=None):
    rclpy.init(args=args)

    node = RGBDReplayNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()