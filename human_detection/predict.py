import glob
import os

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rione_msgs.msg import PredictResult
import joblib

from lib import Logger
from lib.module import numerical_sort

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log/")
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640


class HumanDetectionPredict(Node):

    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.count_complete = 0
        self.log_image_files = None
        self.log_xyz_files = None
        self.log_odom_files = None
        self.face_dataset = []
        self.target_index = 0
        self.create_subscription(String, "/human_detection/command", self.callback_command, 10)
        self.create_subscription(PredictResult, "/face_predictor/result", self.callback_face_predict_result, 10)
        self.pub_human_detection_command = self.create_publisher(String, "/human_detection/command", 10)
        self.pub_image = self.create_publisher(Image, "/face_predictor/color/image", 10)
        self.logger = Logger.Logger(os.path.join(LOG_DIR, "predict"))

    def complete_predict(self):
        if self.count_complete < 3:
            return
        self.logger.save(self.face_dataset)
        self.pub_human_detection_command.publish(String(data="calculation"))

    def callback_face_predict_result(self, msg: PredictResult):
        """
        顔検出の結果受け取り＆次の画像送信
        :param msg:
        :return:
        """
        if not len(msg.point1) == 0:
            # imageとの時間的な連結
            applicable_xyz_index = np.where(self.log_xyz_files[:, 0] > self.log_image_files[self.target_index][0])[0][0]
            applicable_odom_index = \
                np.where(self.log_odom_files[:, 0] > self.log_image_files[self.target_index][0])[0][0]

            # Todo 補完
            for p1, p2 in zip(msg.point1, msg.point2):
                image = np.reshape(self.log_image_files[self.target_index][1], (IMAGE_HEIGHT, IMAGE_WIDTH, 3))
                x = np.nanmean(self.log_xyz_files[applicable_xyz_index][1][0][int(p1.y):int(p2.y), int(p1.x):int(p2.x)])
                y = np.nanmean(self.log_xyz_files[applicable_xyz_index][1][1][int(p1.y):int(p2.y), int(p1.x):int(p2.x)])
                z = np.nanmean(self.log_xyz_files[applicable_xyz_index][1][2][int(p1.y):int(p2.y), int(p1.x):int(p2.x)])
                pos_x = self.log_odom_files[applicable_odom_index][1][0]
                pos_y = self.log_odom_files[applicable_odom_index][1][1]
                pos_z = self.log_odom_files[applicable_odom_index][1][2]
                radian = self.log_odom_files[applicable_odom_index][1][3]
                if not np.isnan(x) and not np.isnan(x) and not np.isnan(x):
                    self.face_dataset.append({
                        "face_image": image[int(p1.y):int(p2.y), int(p1.x):int(p2.x)],
                        "x": x,
                        "y": y,
                        "z": z,
                        "radian": radian,
                        "pos_x": pos_x,
                        "pos_y": pos_y,
                        "pos_z": pos_z,
                    })

        self.target_index = self.target_index + 1
        if self.target_index < len(self.log_image_files):
            self.pub_image.publish(Image(data=self.log_image_files[self.target_index][1]))
        else:
            print("finish")
            self.count_complete = 4
            self.complete_predict()

    def callback_command(self, msg: String):
        """
        顔検出の開始
        :param msg:
        :return:
        """
        if not msg.data == "predict":
            return
        print("Loading...", flush=True)
        image_list = []
        xyz_list = []
        odom_list = []

        # logファイルの読み込み
        for filename in sorted(glob.glob("{}/scan/image/*".format(LOG_DIR)), key=numerical_sort):
            image_list.append(joblib.load(filename))
        self.log_image_files = np.asarray(image_list)

        for filename in sorted(glob.glob("{}/scan/xyz/*".format(LOG_DIR)), key=numerical_sort):
            xyz_list.append(joblib.load(filename))
        self.log_xyz_files = np.asarray(xyz_list)

        for filename in sorted(glob.glob("{}/scan/odometry/*".format(LOG_DIR)), key=numerical_sort):
            odom_list.append(joblib.load(filename))
        self.log_odom_files = np.asarray(odom_list)

        self.pub_image.publish(Image(data=self.log_image_files[0][1]))


def main():
    rclpy.init()
    node = HumanDetectionPredict("HumanDetectionPredict")
    rclpy.spin(node)


if __name__ == '__main__':
    main()
