import os
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import Odometry

from lib import Logger
from lib.module import to_quaternion_rad

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log/")


class HumanDetectionScanOdometry(Node):

    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.is_start = False
        self.count_files = 0
        self.create_subscription(String, "/human_detection/command/scan", self.callback_command, 50)
        self.create_subscription(Odometry, "/turtlebot2/odometry", self.callback_odometry, 1)
        self.logger = Logger.Logger(os.path.join(LOG_DIR, "scan", "odometry"))

    def callback_command(self, msg: String):
        if msg.data == "odometry":
            self.is_start = True
            print("odometryデータ取得開始", flush=True)
        elif msg.data == "stop":
            self.is_start = False

    def callback_odometry(self, msg: Odometry):
        if not self.is_start:
            return
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        radian = to_quaternion_rad(msg.pose.pose.orientation.w, msg.pose.pose.orientation.z)
        odometry = [x, y, z, radian]
        self.logger.save([time.time(), odometry])


def main():
    rclpy.init()
    node = HumanDetectionScanOdometry("HumanDetectionScanOdometry")
    rclpy.spin(node)


if __name__ == '__main__':
    main()
