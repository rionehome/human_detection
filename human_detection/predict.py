import glob
import os
import re
import time

import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import numpy as np
import joblib

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log/")


class HumanDetectionPredict(Node):

    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.around_info_stack = None
        self.is_start = False
        self.color_image = None
        self.create_subscription(
            String,
            "/human_detection/command",
            self.callback_command,
            10
        )

    @staticmethod
    def numerical_sort(value):
        numbers = re.compile(r'(\d+)')
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    def callback_command(self, msg):
        if msg.data == "predict":
            self.is_start = True
        else:
            self.is_start = False
            return
        print("load logs")
        # logファイルの読み込み
        logs = [joblib.load(filename)
                for filename in sorted(glob.glob("{}/scan_*".format(LOG_DIR)), key=self.numerical_sort)]
        self.around_info_stack = np.vstack(logs)
        for around_info in self.around_info_stack:
            cv2.imshow("image", around_info[1])
            cv2.waitKey(1)
            time.sleep(0.034)


def main():
    rclpy.init()
    node = HumanDetectionPredict("HumanDetectionPredict")
    rclpy.spin(node)


if __name__ == '__main__':
    main()
