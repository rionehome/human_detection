import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2
import numpy as np


class HumanDetectionPredict(Node):

    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.is_start = False
        self.color_image = None
        self.create_subscription(String, "/human_detection/command", self.callback_command, 10)

    def callback_command(self, msg):
        if msg.data == "predict":
            self.is_start = True
        else:
            self.is_start = False
            return


def main():
    rclpy.init()
    node = HumanDetectionPredict("HumanDetectionPredict")
    rclpy.spin(node)


if __name__ == '__main__':
    main()
