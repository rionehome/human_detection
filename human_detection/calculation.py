import glob
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import joblib

from lib.module import show_image_tile, calc_real_position

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log/")


class HumanDetectionCalculation(Node):

    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.create_subscription(String, "/human_detection/command", self.callback_command, 10)

    def callback_command(self, msg):
        if not msg.data == "calculation":
            return
        print("Loading...", flush=True)
        # logファイルの読み込み
        face_dataset = joblib.load(glob.glob("{}/predict/*".format(LOG_DIR))[0])
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        face_images = []
        for face_info in face_dataset:
            face_images.append(face_info["face_image"])
            cv2.imshow("color", face_info["face_image"][:, :, [2, 1, 0]])
            cv2.waitKey(1)
            real_pos = calc_real_position(
                face_info["x"],
                face_info["y"],
                face_info["z"],
                face_info["pos_x"],
                face_info["pos_y"],
                face_info["radian"]
            )
            print(real_pos)
            time.sleep(1)
        # print(np.asarray(face_images).shape)
        # cv2.imshow("window", face_info["face_image"])
        # cv2.waitKey(1)
        # ax.scatter(real_pos[0], real_pos[1], real_pos[2])
        # plt.xlim([-5, 5])
        # plt.ylim([-5, 5])
        # show_image_tile(face_images)
        # plt.show()


def main():
    rclpy.init()
    node = HumanDetectionCalculation("HumanDetectionCalculation")
    rclpy.spin(node)


if __name__ == '__main__':
    main()
