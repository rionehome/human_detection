import os
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
import numpy as np

from lib import Logger

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log/")


class HumanDetectionScanXYZ(Node):

    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.is_start = False
        self.count_files = 0
        self.create_subscription(String, "/human_detection/command/scan", self.callback_command, 10)
        self.create_subscription(
            PointCloud2, "/camera/aligned_depth_to_color/color/points", self.callback_point_cloud, 1)
        self.logger = Logger.Logger(os.path.join(LOG_DIR, "scan", "xyz"))

    def callback_command(self, msg: String):
        if msg.data == "xyz":
            self.logger.clear()
            self.is_start = True
            print("xyzデータ取得開始", flush=True)
        elif msg.data == "stop":
            self.is_start = False

    def callback_point_cloud(self, msg: PointCloud2):
        if not self.is_start:
            return
        real_data = np.asarray(msg.data, dtype=np.uint8).view(dtype=np.float32).reshape((msg.height, msg.width, 8))
        self.logger.save([time.time(), [real_data[:, :, 0], real_data[:, :, 1], real_data[:, :, 2]]])


def main():
    rclpy.init()
    node = HumanDetectionScanXYZ("HumanDetectionScanXYZ")
    rclpy.spin(node)


if __name__ == '__main__':
    main()
