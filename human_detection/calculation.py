import glob
import os
import sys

import cv2
import pandas as pd
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from lib.module import calc_real_position, compare, show_image_tile

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log/")
IMAGE_SIZE = 50


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
        # face_images = np.zeros((len(face_dataset), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        train = []
        for i, face_info in enumerate(face_dataset):
            # face_images[i] = normalize_image(face_info["face_image"], IMAGE_SIZE)
            train.append(cv2.resize(face_info["face_image"], (IMAGE_SIZE, IMAGE_SIZE), cv2.INTER_LINEAR))
            real_pos = calc_real_position(
                face_info["x"],
                face_info["y"],
                face_info["z"],
                face_info["pos_x"],
                face_info["pos_y"],
                face_info["radian"]
            )
            print(real_pos)
        # ax.scatter(real_pos[0], real_pos[1], real_pos[2])
        # plt.xlim([-5, 5])
        # plt.ylim([-5, 5])
        # plt.show()
        distances = np.zeros((len(train), len(train)))
        for i, img in enumerate(train):
            distances[i, :] = [compare(img, f) for f in train]
        plt.clf()
        plt.hist(distances.flatten(), bins=50)
        plt.title('Histogram of distance matrix')
        plt.show()
        cls = DBSCAN(metric='precomputed', min_samples=5, eps=0.9)
        y = cls.fit_predict(distances)
        show_image_tile([np.array(train)[y == uniq] for uniq in pd.Series(y).value_counts().index])
        sys.exit(0)


def main():
    rclpy.init()
    node = HumanDetectionCalculation("HumanDetectionCalculation")
    rclpy.spin(node)


if __name__ == '__main__':
    main()
