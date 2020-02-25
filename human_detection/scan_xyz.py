import os
import shutil

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
import numpy as np
import joblib

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log/")


class HumanDetectionScanXYZ(Node):

    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.is_start = False
        self.count_files = 0
        self.create_subscription(
            String,
            "/human_detection/command",
            self.callback_command,
            10
        )
        self.create_subscription(
            PointCloud2,
            "/camera/aligned_depth_to_color/color/points",
            self.callback_point_cloud,
            10
        )

    def save(self, save_data: list, typename: str):
        if not os.path.exists(LOG_DIR):  # ディレクトリがなければ
            os.makedirs(LOG_DIR)
        joblib.dump(save_data, os.path.join(LOG_DIR, "scan_{}.{}.npy".format(typename, self.count_files + 1)),
                    compress=True)
        # np.save(os.path.join(LOG_DIR, "scan_{}.{}.npy".format(typename, self.count_files + 1)), np.asarray(save_data))
        save_data.clear()
        self.count_files = self.count_files + 1
        # print("save {}".format(typename))

    def callback_command(self, msg: String):
        if msg.data == "start":
            self.is_start = True
        else:
            self.is_start = False
            return
        if os.path.exists(LOG_DIR):  # ディレクトリがあれば
            shutil.rmtree(LOG_DIR)
            os.makedirs(LOG_DIR)
        print("データ取得開始")

    def callback_point_cloud(self, msg: PointCloud2):
        if not self.is_start:
            return
        real_data = np.asarray(msg.data, dtype=np.uint8).view(dtype=np.float32).reshape((msg.height, msg.width, 8))
        point_xyz = [real_data[:, :, 0], real_data[:, :, 1], real_data[:, :, 2]]
        # cv2.imshow("depth", (self.point_xyz[2] * 25).astype(int).astype(np.uint8))
        # cv2.waitKey(1)


def main():
    rclpy.init()
    node = HumanDetectionScanXYZ("HumanDetectionScanXYZ")
    rclpy.spin(node)


if __name__ == '__main__':
    main()
