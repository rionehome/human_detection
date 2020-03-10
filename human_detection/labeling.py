import glob
import os

import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import joblib

from lib.module import calc_real_position

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log/")
SAMPLE_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_image/")
IMAGE_SIZE = 96
LABEL_COLOR_SET = {-1: "black", 0: "red", 1: "green", 2: "blue", 3: "yellow", 4: "purple", 5: "blown"}


class HumanDetectionLabeling(Node):

    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.create_subscription(String, "/human_detection/command", self.callback_command, 10)
        self.real_positions = []
        self.sampled_imgs = []

    def callback_command(self, msg):
        if not msg.data == "labeling":
            return
        print("Loading...", flush=True)
        # logファイルの読み込み
        face_logs = joblib.load(glob.glob("{}/calculation/*".format(LOG_DIR))[0])

        for log in face_logs:
            self.sampled_imgs.append(cv2.resize(log["face_image"], (IMAGE_SIZE, IMAGE_SIZE), cv2.INTER_LINEAR))
            self.real_positions.append(calc_real_position(
                log["x"],
                log["y"],
                log["z"],
                log["pos_x"],
                log["pos_y"],
                log["radian"]
            ))


def main():
    rclpy.init()
    node = HumanDetectionLabeling("HumanDetectionLabeling")
    rclpy.spin(node)


if __name__ == '__main__':
    main()
