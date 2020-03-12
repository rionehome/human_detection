import glob
import os
import sys

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN

from lib import Logger
from lib.module import calc_real_position, compare_point, compare_image, show_image_tile

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log/")
IMAGE_SIZE = 96
LABEL_COLOR_SET = {-1: "black", 0: "red", 1: "green", 2: "blue", 3: "yellow", 4: "purple", 5: "blown"}


class HumanDetectionCalculation(Node):

    def __init__(self, node_name: str):
        np.set_printoptions(suppress=True)
        super().__init__(node_name)
        self.create_subscription(String, "/human_detection/command", self.callback_command, 10)
        self.pub_command = self.create_publisher(String, "/human_detection/command", 10)
        self.real_positions = []
        self.sampled_imgs = []
        self.logger = Logger.Logger(os.path.join(LOG_DIR, "calculation"))

    def callback_command(self, msg):
        if not msg.data == "calculation":
            return
        self.logger.clear()
        print("Loading...", flush=True)
        # logファイルの読み込み
        face_logs = joblib.load(glob.glob("{}/sampling/*".format(LOG_DIR))[0])

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

        num_sample = len(self.sampled_imgs)

        # dbscan下準備
        distance_matrix = np.zeros((num_sample, num_sample))
        for row in range(num_sample):
            distance_matrix[row, :] = [
                compare_point(
                    self.real_positions[row],
                    self.real_positions[col]
                ) * 0.1 + compare_image(
                    self.sampled_imgs[row],
                    self.sampled_imgs[col]
                ) for col in range(num_sample)
            ]

        plt.clf()
        plt.hist(distance_matrix.flatten(), bins=50)
        plt.title('Histogram of distance matrix')
        plt.show()

        cls = DBSCAN(metric='precomputed', min_samples=4, eps=0.4)
        labels = cls.fit_predict(distance_matrix)

        face_infos = []
        for uniq in np.unique(labels):
            if uniq == -1:
                continue
            average_point = np.average(np.array(self.real_positions)[labels == uniq], axis=0)
            face_infos.append({
                "face_image": np.array(self.sampled_imgs)[labels == uniq],
                "position": average_point
            })
            print(average_point, flush=True)
        print("finish", flush=True)
        self.logger.save(face_infos)
        self.pub_command.publish(String(data="labeling"))

        # 描画
        for uniq in np.unique(labels):
            show_image_tile([np.array(self.sampled_imgs)[labels == uniq]], title="label: " + str(uniq))

        fig = plt.figure()
        ax = Axes3D(fig)
        for i in range(num_sample):
            ax.scatter(self.real_positions[i][0], self.real_positions[i][1], self.real_positions[i][2])
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-3, 3)
        plt.show()
        plt.clf()

        fig = plt.figure()
        ax = Axes3D(fig)
        for uniq in np.unique(labels):
            face_points = np.array(self.real_positions)[labels == uniq]
            average_point = np.average(face_points, axis=0)
            ax.scatter(face_points[:, 0], face_points[:, 1], face_points[:, 2],
                       color=LABEL_COLOR_SET[-1 if uniq == -1 else uniq % 5])
            if uniq == -1:
                continue
            ax.scatter(average_point[0], average_point[1], average_point[2], marker="x", s=300,
                       color=LABEL_COLOR_SET[-1 if uniq == -1 else uniq % 5])
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-3, 3)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()
        sys.exit(0)


def main():
    rclpy.init()
    node = HumanDetectionCalculation("HumanDetectionCalculation")
    rclpy.spin(node)


if __name__ == '__main__':
    main()
