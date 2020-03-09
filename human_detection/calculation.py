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
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN

from lib.module import calc_real_position, compare, show_image_tile

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log/")
IMAGE_SIZE = 50
LABEL_COLOR_SET = {-1: "black", 0: "red", 1: "green", 2: "blue", 3: "yellow", 4: "purple", 5: "blown"}


class HumanDetectionCalculation(Node):

    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.create_subscription(String, "/human_detection/command", self.callback_command, 10)
        self.real_positions = []
        self.face_imgs = []

    def callback_command(self, msg):
        if not msg.data == "calculation":
            return
        print("Loading...", flush=True)
        # logファイルの読み込み
        face_logs = joblib.load(glob.glob("{}/predict/*".format(LOG_DIR))[0])
        num_logs = len(face_logs)

        for log in face_logs:
            self.face_imgs.append(cv2.resize(log["face_image"], (IMAGE_SIZE, IMAGE_SIZE), cv2.INTER_LINEAR))
            self.real_positions.append(calc_real_position(
                log["x"],
                log["y"],
                log["z"],
                log["pos_x"],
                log["pos_y"],
                log["radian"]
            ))

        # クラスタリング下準備
        distance_matrix = np.zeros((num_logs, num_logs))
        for row in range(num_logs):
            distance_matrix[row, :] = [
                compare(
                    self.face_imgs[row],
                    self.real_positions[row],
                    self.face_imgs[col],
                    self.real_positions[col]
                ) for col in range(num_logs)
            ]

        # """
        plt.clf()
        plt.hist(distance_matrix.flatten(), bins=50)
        plt.title('Histogram of distance matrix')
        plt.show()
        # """
        cls = DBSCAN(metric='precomputed', min_samples=5, eps=1.5)
        labels = cls.fit_predict(distance_matrix)
        for uniq in pd.Series(labels).value_counts().index:
            print(uniq)
            show_image_tile([np.array(self.face_imgs)[labels == uniq]])

        # fig = plt.figure()
        # ax = Axes3D(fig)
        # for i in range(num_logs):
        #    ax.scatter(self.real_positions[i][0], self.real_positions[i][1], self.real_positions[i][2])
        # plt.xlim([-5, 5])
        # plt.ylim([-5, 5])
        # plt.show()
        # plt.clf()

        fig = plt.figure()
        ax = Axes3D(fig)
        for i in range(num_logs):
            if labels[i] == -1 and not len(set(labels)) == 1:
                continue
            ax.scatter(self.real_positions[i][0], self.real_positions[i][1], self.real_positions[i][2],
                       color=LABEL_COLOR_SET[labels[i]])
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-3, 3)
        plt.show()
        sys.exit(0)


def main():
    rclpy.init()
    node = HumanDetectionCalculation("HumanDetectionCalculation")
    rclpy.spin(node)


if __name__ == '__main__':
    main()
