import os
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image

from lib import Logger

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")


class HumanDetectionScanImage(Node):

    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.is_start = False
        self.count_files = 0
        self.create_subscription(String, "/human_detection/command/scan", self.callback_command, 50)
        self.create_subscription(Image, "/camera/color/image_raw", self.callback_color_image, 1)
        self.logger = Logger.Logger(os.path.join(LOG_DIR, "scan", "image"))

    def callback_command(self, msg: String):
        if msg.data == "image":
            self.is_start = True
            print("imageデータ取得開始", flush=True)
        elif msg.data == "stop":
            self.is_start = False

    def callback_color_image(self, msg: Image):
        if not self.is_start:
            return
        self.logger.save([time.time(), msg.data])


def main():
    rclpy.init()
    node = HumanDetectionScanImage("HumanDetectionScanImage")
    rclpy.spin(node)


if __name__ == '__main__':
    main()
