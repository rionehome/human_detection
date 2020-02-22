import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2
import numpy as np


class HumanDetectionScan(Node):

    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.is_start = False
        self.color_image = None
        self.point_xyz = None
        self.stack_around_info = []
        self.create_subscription(
            String,
            "/human_detection/command",
            self.callback_command,
            10
        )
        self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.callback_color_image,
            10
        )
        self.create_subscription(
            PointCloud2,
            "/camera/aligned_depth_to_color/color/points",
            self.callback_point_cloud,
            10
        )

    def callback_command(self, msg):
        if msg.data == "start":
            self.is_start = True
        else:
            self.is_start = False
            return

        while rclpy.ok():
            pass

    def callback_odom(self, msg):
        if not self.is_start:
            return
        pass

    def callback_color_image(self, msg: Image):
        if not self.is_start:
            return

        self.color_image = np.asarray(msg.data).reshape((msg.height, msg.width, 3)).astype(np.uint8)
        cv2.imshow("depth", self.color_image)
        cv2.waitKey(1)

    def callback_point_cloud(self, msg: PointCloud2):
        if not self.is_start:
            return

        real_data = np.asarray(msg.data, dtype=np.uint8).view(dtype=np.float32).reshape((msg.height, msg.width, 8))
        self.point_xyz = np.asarray([real_data[:, :, 0], real_data[:, :, 1], real_data[:, :, 2]])

        cv2.imshow("depth", (real_data[:, :, 2] * 25).astype(int).astype(np.uint8))
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = HumanDetectionScan("HumanDetectionScan")
    rclpy.spin(node)


if __name__ == '__main__':
    main()
