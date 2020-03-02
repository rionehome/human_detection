import glob
import os
import re
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rione_msgs.msg import PredictResult
import joblib

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log/")


class HumanDetectionPredict(Node):

    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.log_image_files = None
        self.log_xyz_files = None
        self.target_index = 0
        self.create_subscription(String, "/human_detection/command", self.callback_command, 10)
        self.create_subscription(PredictResult, "/face_predictor/result", self.callback_face_predict_result, 10)
        self.pub_image = self.create_publisher(Image, "/face_predictor/color/image", 10)

    @staticmethod
    def numerical_sort(value):
        numbers = re.compile(r'(\d+)')
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    def callback_face_predict_result(self, msg: PredictResult):
        """
        顔検出の結果受け取り＆次の画像送信
        :param msg:
        :return:
        """
        self.target_index = self.target_index + 1
        if not len(msg.point1) == 0:
            # imageとxyz画像の時間的な連結
            applicable_index = np.where(self.log_xyz_files[:, 0] > self.log_image_files[self.target_index - 1][0])[0][0]
            # Todo 補完
            print(applicable_index)

        if self.target_index < len(self.log_image_files):
            self.pub_image.publish(Image(data=self.log_image_files[self.target_index][1]))
        else:
            print("finish")

    def callback_command(self, msg: String):
        """
        顔検出の開始
        :param msg:
        :return:
        """
        if not msg.data == "predict":
            return
        print("load logs", flush=True)
        image_list = []
        xyz_list = []
        # logファイルの読み込み
        for filename in sorted(glob.glob("{}/image/scan_*".format(LOG_DIR)), key=self.numerical_sort):
            image_list.append(joblib.load(filename))
        self.log_image_files = np.asarray(image_list)

        for filename in sorted(glob.glob("{}/xyz/scan_*".format(LOG_DIR)), key=self.numerical_sort):
            xyz_list.append(joblib.load(filename))
        self.log_xyz_files = np.asarray(xyz_list)

        self.pub_image.publish(Image(data=self.log_image_files[0][1]))


def main():
    rclpy.init()
    node = HumanDetectionPredict("HumanDetectionPredict")
    rclpy.spin(node)


if __name__ == '__main__':
    main()
