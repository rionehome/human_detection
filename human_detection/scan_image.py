import os
import shutil

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
import numpy as np
import joblib

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log/")


class HumanDetectionScanImage(Node):

    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.is_start = False
        self.count_files = 0
        self.create_subscription(
            String,
            "/human_detection/command/scan",
            self.callback_command,
            10
        )
        self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.callback_color_image,
            10
        )

    def save(self, save_data: list, typename: str):
        if not os.path.exists(LOG_DIR):  # ディレクトリがなければ
            os.makedirs(LOG_DIR)
        joblib.dump(save_data, os.path.join(LOG_DIR, "scan_{}.{}.npy".format(typename, self.count_files + 1)),
                    compress=True)
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

    def callback_color_image(self, msg: Image):
        if not self.is_start:
            return
        color_image = np.asarray(msg.data).reshape((msg.height, msg.width, 3)).astype(np.uint8)
        # cv2.imshow("color", self.color_image)
        # cv2.waitKey(1)


def main():
    rclpy.init()
    node = HumanDetectionScanImage("HumanDetectionScanImage")
    rclpy.spin(node)


if __name__ == '__main__':
    main()
