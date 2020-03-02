import os
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
import joblib
import numpy as np

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log/")


class HumanDetectionScanXYZ(Node):

    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.is_start = False
        self.count_files = 0
        self.create_subscription(String, "/human_detection/command/scan", self.callback_command, 50)
        self.create_subscription(
            PointCloud2, "/camera/aligned_depth_to_color/color/points", self.callback_point_cloud, 1)

    def save(self, save_data: list, typename: str):
        save_path = os.path.join(LOG_DIR, typename)
        if not os.path.exists(save_path):  # ディレクトリがなければ
            os.makedirs(save_path)
        joblib.dump(save_data, os.path.join(save_path, "scan_{}.{}.npy".format(typename, self.count_files + 1)),
                    compress=True)
        save_data.clear()
        self.count_files = self.count_files + 1

    def callback_command(self, msg: String):
        if msg.data == "xyz":
            self.is_start = True
            print("xyzデータ取得開始", flush=True)
        elif msg.data == "stop":
            self.is_start = False

    def callback_point_cloud(self, msg: PointCloud2):
        if not self.is_start:
            return
        real_data = np.asarray(msg.data, dtype=np.uint8).view(dtype=np.float32).reshape((msg.height, msg.width, 8))
        self.save([time.time(), [real_data[:, :, 0], real_data[:, :, 1], real_data[:, :, 2]]], "xyz")


def main():
    rclpy.init()
    node = HumanDetectionScanXYZ("HumanDetectionScanXYZ")
    rclpy.spin(node)


if __name__ == '__main__':
    main()
