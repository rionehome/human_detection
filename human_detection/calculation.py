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
from sklearn.cluster import KMeans

from lib import Logger
from lib.module import calc_real_position, compare_point, compare_image, show_image_tile

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log/")
SAMPLE_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_image/")
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

        # ノイズの挿入
        self.sampled_imgs.append(
            cv2.cvtColor(cv2.imread(os.path.join(SAMPLE_IMAGE_PATH, "not_person.png")), cv2.COLOR_BGR2RGB)
        )
        self.real_positions.append((0, 0, 0))

        num_sample = len(self.sampled_imgs)

        # k-means下準備
        distance_matrix = np.zeros((num_sample, num_sample))
        for row in range(num_sample):
            distance_matrix[row, :] = [
                compare_image(
                    self.sampled_imgs[row],
                    self.sampled_imgs[col],
                ) for col in range(num_sample)
            ]

        cls = KMeans(n_clusters=2)
        labels = cls.fit_predict(distance_matrix)
        show_image_tile([np.array(self.sampled_imgs)[labels != labels[-1]]], title="face")
        show_image_tile([np.array(self.sampled_imgs)[labels == labels[-1]]], title="not_face")

        # 物体の排除
        face_imgs = np.array(self.sampled_imgs)[labels != labels[-1]]
        face_real_positions = np.array(self.real_positions)[labels != labels[-1]]
        num_face = face_real_positions.shape[0]
        # self.sampled_imgs.pop(-1)  # 人為的ノイズの削除

        # dbscanグ下準備
        distance_matrix = np.zeros((num_face, num_face))
        for row in range(num_face):
            distance_matrix[row, :] = [
                compare_point(
                    face_real_positions[row],
                    face_real_positions[col]
                ) for col in range(num_face)
            ]

        plt.clf()
        plt.hist(distance_matrix.flatten(), bins=50)
        plt.title('Histogram of distance matrix')
        plt.show()

        cls = DBSCAN(metric='precomputed', min_samples=5, eps=0.5)
        face_labels = cls.fit_predict(distance_matrix)

        face_infos = []
        for uniq in np.unique(face_labels):
            if uniq == -1:
                continue
            average_point = np.average(np.array(face_real_positions)[face_labels == uniq], axis=0)
            face_infos.append({
                "face_image": np.array(face_imgs)[face_labels == uniq],
                "position": average_point
            })
            print(average_point, flush=True)
        print("finish", flush=True)
        self.logger.save(face_infos)
        self.pub_command.publish(String(data="labeling"))

        # 描画
        for uniq in np.unique(face_labels):
            show_image_tile([np.array(face_imgs)[face_labels == uniq]], title="label: " + str(uniq))

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
        for uniq in np.unique(face_labels):
            face_points = np.array(face_real_positions)[face_labels == uniq]
            average_point = np.average(face_points, axis=0)
            ax.scatter(face_points[:, 0], face_points[:, 1], face_points[:, 2],
                       color=LABEL_COLOR_SET[-1 if uniq == -1 else uniq % 5])
            ax.scatter(average_point[0], average_point[1], average_point[2], marker="x", s=300,
                       color=LABEL_COLOR_SET[-1 if uniq == -1 else uniq % 5])
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
