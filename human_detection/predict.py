import glob
import os
import re

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rione_msgs.msg import PredictResult
import numpy as np
import joblib

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log/")


class HumanDetectionPredict(Node):

    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.cv_bridge = CvBridge()
        self.around_info_stack = None
        self.is_start = False
        self.target_index = 0
        self.create_subscription(String, "/human_detection/command", self.callback_command, 10)
        self.create_subscription(PredictResult, "/face_predictor/result", self.callback_result, 10)
        self.pub_image = self.create_publisher(Image, "/face_predictor/color/image", 10)

    @staticmethod
    def numerical_sort(value):
        numbers = re.compile(r'(\d+)')
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    def callback_result(self, msg: PredictResult):
        self.target_index = self.target_index + 1
        print(msg)
        if self.target_index < self.around_info_stack.shape[0]:
            self.pub_image.publish(self.cv_bridge.cv2_to_imgmsg(self.around_info_stack[self.target_index][1]))

    def callback_command(self, msg: String):
        if msg.data == "predict":
            self.is_start = True
        else:
            self.is_start = False
            return
        print("load logs")
        # logファイルの読み込み
        logs = [joblib.load(filename)
                for filename in sorted(glob.glob("{}/scan_*".format(LOG_DIR)), key=self.numerical_sort)]
        self.around_info_stack = np.vstack(logs)
        self.pub_image.publish(self.cv_bridge.cv2_to_imgmsg(self.around_info_stack[0][1]))


def main():
    rclpy.init()
    node = HumanDetectionPredict("HumanDetectionPredict")
    rclpy.spin(node)


if __name__ == '__main__':
    main()
