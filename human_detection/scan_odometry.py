import os
import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import Odometry
import numpy as np
import joblib

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log/")


class HumanDetectionScanOdometry(Node):

    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.is_start = False
        self.count_files = 0
        self.create_subscription(String, "/human_detection/command/scan", self.callback_command, 50)
        self.create_subscription(Odometry, "/turtlebot2/odometry", self.callback_odometry, 1)

    @staticmethod
    def to_quaternion_rad(w, z):
        return math.acos(w) * 2 * np.sign(z)

    def save(self, save_data: list, typename: str):
        save_path = os.path.join(LOG_DIR, typename)
        if not os.path.exists(save_path):  # ディレクトリがなければ
            os.makedirs(save_path)
        joblib.dump(save_data, os.path.join(save_path, "scan_{}.{}.npy".format(typename, self.count_files + 1)),
                    compress=True)
        save_data.clear()
        self.count_files = self.count_files + 1

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
        radian = self.to_quaternion_rad(msg.pose.pose.orientation.w, msg.pose.pose.orientation.z)
        odometry = [x, y, z, radian]
        self.save([time.time(), odometry], "odometry")


def main():
    rclpy.init()
    node = HumanDetectionScanOdometry("HumanDetectionScanOdometry")
    rclpy.spin(node)


if __name__ == '__main__':
    main()
