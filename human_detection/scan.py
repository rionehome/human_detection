import os

# import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from rione_msgs.msg import Command
from sensor_msgs.msg import Image, PointCloud2
import numpy as np
import math
import joblib

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log/")
NUM_SAVE_POINT = 1


class HumanDetectionScan(Node):

    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.is_start = False
        self.count_files = 0
        self.around_info_stack = []
        self.color_image = None
        self.point_xyz = None
        self.odometry = None
        self.pub_turn_command = self.create_publisher(
            Command,
            "/turn_robot/command",
            10
        )
        self.create_subscription(
            String,
            "/human_detection/command",
            self.callback_command,
            10
        )
        self.create_subscription(
            Odometry,
            "localization",
            self.callback_odometry,
            10
        )
        self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.callback_color_image,
            10
        )
        self.create_subscription(
            PointCloud2,
            "/camera/aligned_depth_to_color/color/points",
            self.callback_point_cloud,
            10
        )
        self.create_subscription(
            String,
            "/turn_robot/status",
            self.callback_turn_status,
            10
        )
        self.pub_human_detection_command = self.create_publisher(
            String,
            "/human_detection/command",
            10
        )

    @staticmethod
    def to_quaternion_rad(w, z):
        return math.acos(w) * 2 * np.sign(z)

    def save(self, save_data: list, typename: str):
        if not os.path.exists(LOG_DIR):  # ディレクトリがなければ
            os.makedirs(LOG_DIR)
        joblib.dump(save_data, os.path.join(LOG_DIR, "scan_{}.{}.npy".format(typename, self.count_files + 1)),
                    compress=True)
        # np.save(os.path.join(LOG_DIR, "scan_{}.{}.npy".format(typename, self.count_files + 1)), np.asarray(save_data))
        save_data.clear()
        self.count_files = self.count_files + 1
        # print("save {}".format(typename))

    def callback_command(self, msg: String):
        if msg.data == "start":
            self.is_start = True
        else:
            self.is_start = False
            return
        # 回転の開始
        self.pub_turn_command.publish(Command(command="START", content="360"))
        print("データ取得開始")

    def callback_odometry(self, msg: Odometry):
        if not self.is_start:
            return
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        radian = self.to_quaternion_rad(msg.pose.pose.orientation.w, msg.pose.pose.orientation.z)
        self.odometry = [x, y, z, radian]

    def callback_color_image(self, msg: Image):
        if not self.is_start:
            return
        self.color_image = np.asarray(msg.data).reshape((msg.height, msg.width, 3)).astype(np.uint8)
        # cv2.imshow("color", self.color_image)
        # cv2.waitKey(1)

    def callback_point_cloud(self, msg: PointCloud2):
        if not self.is_start:
            return
        real_data = np.asarray(msg.data, dtype=np.uint8).view(dtype=np.float32).reshape((msg.height, msg.width, 8))
        self.point_xyz = [real_data[:, :, 0], real_data[:, :, 1], real_data[:, :, 2]]
        # cv2.imshow("depth", (self.point_xyz[2] * 25).astype(int).astype(np.uint8))
        # cv2.waitKey(1)

        # データの統合
        if self.color_image is None or self.point_xyz is None:
            return
        self.save([self.odometry, self.color_image, self.point_xyz], "around_info")

    def callback_turn_status(self, msg: String):
        if not self.is_start or not msg.data == "FINISH":
            return
        self.is_start = False
        self.save([self.odometry, self.color_image, self.point_xyz], "around_info")
        print("scanデータ保存完了")
        self.pub_human_detection_command.publish(String(data="predict"))


def main():
    rclpy.init()
    node = HumanDetectionScan("HumanDetectionScan")
    rclpy.spin(node)


if __name__ == '__main__':
    main()
