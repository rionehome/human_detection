import glob
import math
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import joblib

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log/")


class HumanDetectionCalculation(Node):

    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.create_subscription(String, "/human_detection/command", self.callback_command, 10)

    def calc_real_position(self, x, y, z, pos_x, pos_y, pos_radian):
        relative_x = z
        relative_y = -x
        relative_z = y
        result_x = (relative_x * math.cos(pos_radian) - relative_y * math.sin(pos_radian)) + pos_x
        result_y = (relative_x * math.sin(pos_radian) + relative_y * math.cos(pos_radian)) + pos_y
        result_z = relative_z
        return result_x, result_y, result_z

    def callback_command(self, msg):
        if not msg.data == "calculation":
            return
        print("Loading...", flush=True)
        # logファイルの読み込み
        face_dataset = joblib.load(glob.glob("{}/predict/*".format(LOG_DIR))[0])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for face_info in face_dataset:
            real_pos = self.calc_real_position(
                face_info["x"],
                face_info["y"],
                face_info["z"],
                face_info["pos_x"],
                face_info["pos_y"],
                face_info["radian"]
            )
            print(real_pos)
            ax.scatter(real_pos[0], real_pos[1], real_pos[2])
        # plt.xlim([-5, 5])
        # plt.ylim([-5, 5])
        plt.show()


def main():
    rclpy.init()
    node = HumanDetectionCalculation("HumanDetectionCalculation")
    rclpy.spin(node)


if __name__ == '__main__':
    main()
