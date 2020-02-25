import glob
import os
import re

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
        self.log_filenames = []
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
        print(msg)
        if self.target_index < len(self.log_filenames):
            self.pub_image.publish(Image(data=joblib.load(self.log_filenames[self.target_index])[1]))

    def callback_command(self, msg: String):
        """
        顔検出の開始
        :param msg:
        :return:
        """
        if not msg.data == "predict":
            return
        print("load logs")
        # logファイルの読み込み
        self.log_filenames = [filename for filename in
                              sorted(glob.glob("{}/image/scan_*".format(LOG_DIR)), key=self.numerical_sort)]
        self.pub_image.publish(Image(data=joblib.load(self.log_filenames[0])[1]))


def main():
    rclpy.init()
    node = HumanDetectionPredict("HumanDetectionPredict")
    rclpy.spin(node)


if __name__ == '__main__':
    main()
