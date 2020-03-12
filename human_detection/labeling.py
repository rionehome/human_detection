import glob
import os

import cv2
import statistics
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rione_msgs.msg import PredictResult
from sensor_msgs.msg import Image
from std_msgs.msg import String
import joblib

from lib import Logger

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log/")
IMAGE_SIZE = 96


class HumanDetectionLabeling(Node):

    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.create_subscription(String, "/human_detection/command", self.callback_command, 10)
        self.create_subscription(PredictResult, "/gender_predictor/result", self.callback_gender_predict_result, 10)
        self.pub_gender_predictor = self.create_publisher(Image, "/gender_predictor/color/image", 10)
        self.logger = Logger.Logger(os.path.join(LOG_DIR, "labeling"))
        self.bridge = CvBridge()
        self.iterator_indexes = [0, 0]  # 0:face_id, 1:image_index
        self.face_infos = []

    def callback_command(self, msg):
        if not msg.data == "labeling":
            return
        self.logger.clear()
        print("Loading...", flush=True)
        # logファイルの読み込み
        self.face_infos = joblib.load(glob.glob("{}/calculation/*".format(LOG_DIR))[0])

        self.face_infos[self.iterator_indexes[0]].setdefault("genders", [])
        self.pub_gender_predictor.publish(
            self.bridge.cv2_to_imgmsg(
                self.face_infos[self.iterator_indexes[0]]["face_image"][self.iterator_indexes[1]],
                encoding="bgr8"
            )
        )

    def callback_gender_predict_result(self, msg: PredictResult):
        print(msg.class_name.data, flush=True)
        self.face_infos[self.iterator_indexes[0]]["genders"].append(msg.class_name.data)
        if self.iterator_indexes[1] + 1 < len(self.face_infos[self.iterator_indexes[0]]["face_image"]):
            self.iterator_indexes[1] = self.iterator_indexes[1] + 1
        elif self.iterator_indexes[0] + 1 < len(self.face_infos):
            self.iterator_indexes[0] = self.iterator_indexes[0] + 1
            self.iterator_indexes[1] = 0
            self.face_infos[self.iterator_indexes[0]].setdefault("genders", [])
        else:
            print("finish", flush=True)
            for i in range(len(self.face_infos)):
                print(statistics.mode(self.face_infos[i]["genders"]), flush=True)
                human_state = "standing" if self.face_infos[i]["position"][2] > 0.5 else "sitting"
                cv2.imwrite(
                    os.path.join(LOG_DIR, "labeling",
                                 "{}-{}-{}.png".format(i, statistics.mode(self.face_infos[i]["genders"]), human_state)),
                    self.face_infos[i]["face_image"][0][:, :, [2, 1, 0]]
                )
            return

        self.pub_gender_predictor.publish(
            self.bridge.cv2_to_imgmsg(
                self.face_infos[self.iterator_indexes[0]]["face_image"][self.iterator_indexes[1]],
                encoding="bgr8"
            )
        )


def main():
    rclpy.init()
    node = HumanDetectionLabeling("HumanDetectionLabeling")
    rclpy.spin(node)


if __name__ == '__main__':
    main()
