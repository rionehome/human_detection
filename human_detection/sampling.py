import glob
import os

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rione_msgs.msg import PredictResult
import joblib
from cv_bridge import CvBridge
from keras import models

from lib import Logger
from lib.module import numerical_sort, show_image_tile

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log/")
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
INPUT_SIZE = 96
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../human_detection/models/")


class HumanDetectionSampling(Node):

    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.count_complete = 0
        self.log_image_files = None
        self.log_xyz_files = None
        self.log_odom_files = None
        self.face_dataset = []
        self.target_face_index = 0
        self.create_subscription(String, "/human_detection/command", self.callback_command, 10)
        self.create_subscription(PredictResult, "/face_predictor/result", self.callback_face_predict_result, 10)
        self.pub_command = self.create_publisher(String, "/human_detection/command", 10)
        self.pub_face_predictor = self.create_publisher(Image, "/face_predictor/color/image", 10)
        self.logger = Logger.Logger(os.path.join(LOG_DIR, "sampling"))
        self.bridge = CvBridge()
        self.model = models.load_model("{}/judge_face_vgg16.model".format(MODEL_PATH))
        print("load complete!", flush=True)

    def is_face(self, image_array: np.ndarray):
        y = self.model.predict(cv2.resize(image_array, (INPUT_SIZE, INPUT_SIZE))[np.newaxis, :, :, :]).argmax(axis=1)[0]

        return not bool(y)

    def callback_face_predict_result(self, msg: PredictResult):
        """
        顔検出の結果受け取り＆次の画像送信
        :param msg:
        :return:
        """
        if not len(msg.point1) == 0:
            # imageとの時間的な連結
            applicable_xyz_index_array = np.where(
                self.log_xyz_files[:, 0] > self.log_image_files[self.target_face_index][0]
            )
            xyz_index = -1 if applicable_xyz_index_array[0].shape[0] == 0 else applicable_xyz_index_array[0][0]

            applicable_odom_index_array = np.where(
                self.log_odom_files[:, 0] > self.log_image_files[self.target_face_index][0]
            )
            odom_index = -1 if applicable_odom_index_array[0].shape[0] == 0 else applicable_odom_index_array[0][0]

            # Todo 補完
            for p1, p2 in zip(msg.point1, msg.point2):
                image = np.reshape(self.log_image_files[self.target_face_index][1], (IMAGE_HEIGHT, IMAGE_WIDTH, 3))
                x = np.nanmean(self.log_xyz_files[xyz_index][1][0][int((p1.y + p2.y) / 2), int((p1.x + p2.x) / 2)])
                y = np.nanmean(self.log_xyz_files[xyz_index][1][1][int((p1.y + p2.y) / 2), int((p1.x + p2.x) / 2)])
                z = np.nanmean(self.log_xyz_files[xyz_index][1][2][int((p1.y + p2.y) / 2), int((p1.x + p2.x) / 2)])
                pos_x = self.log_odom_files[odom_index][1][0]
                pos_y = self.log_odom_files[odom_index][1][1]
                pos_z = self.log_odom_files[odom_index][1][2]
                radian = self.log_odom_files[odom_index][1][3]
                if not np.isnan(x) and not np.isnan(y) and not np.isnan(z):
                    ex_p1_x = 0 if int(p1.x - 10) < 0 else int(p1.x - 10)
                    ex_p1_y = 0 if int(p1.y - 10) < 0 else int(p1.y - 10)
                    ex_p2_x = IMAGE_WIDTH - 1 if int(p2.x + 10) > IMAGE_WIDTH - 1 else int(p2.x + 10)
                    ex_p2_y = IMAGE_HEIGHT - 1 if int(p2.y + 10) > IMAGE_HEIGHT - 1 else int(p2.y + 10)
                    show_image_tile(images_array=[
                        cv2.resize(image[ex_p1_y:ex_p2_y, ex_p1_x:ex_p2_x].astype(np.uint8), (INPUT_SIZE, INPUT_SIZE))],
                        save_dir=LOG_DIR + "/not_nan/")
                    if self.is_face(image[ex_p1_y:ex_p2_y, ex_p1_x:ex_p2_x]):
                        self.face_dataset.append({
                            "face_image": image[ex_p1_y:ex_p2_y, ex_p1_x:ex_p2_x],
                            "x": x,
                            "y": y,
                            "z": z,
                            "radian": radian,
                            "pos_x": pos_x,
                            "pos_y": pos_y,
                            "pos_z": pos_z,
                        })
                else:
                    show_image_tile(images_array=[self.log_xyz_files[xyz_index][1][2]], save_dir=LOG_DIR + "/nan/")
                    print(np.isnan(x), np.isnan(y), np.isnan(z))

        self.target_face_index = self.target_face_index + 1
        if self.target_face_index < len(self.log_image_files):
            self.pub_face_predictor.publish(Image(data=self.log_image_files[self.target_face_index][1]))
        else:
            print("finish", flush=True)
            self.logger.save(self.face_dataset)
            self.pub_command.publish(String(data="calculation"))

    def callback_command(self, msg: String):
        """
        顔検出の開始
        :param msg:
        :return:
        """
        if not msg.data == "sampling":
            return
        self.logger.clear()
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

        self.pub_face_predictor.publish(Image(data=self.log_image_files[0][1]))


def main():
    rclpy.init()
    node = HumanDetectionSampling("HumanDetectionSampling")
    rclpy.spin(node)


if __name__ == '__main__':
    main()
