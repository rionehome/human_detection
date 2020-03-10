import os
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rione_msgs.msg import Command

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log/")


class HumanDetectionScanMain(Node):

    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.create_subscription(String, "/human_detection/command", self.callback_command, 50)
        self.create_subscription(String, "/turn_robot/status", self.callback_turn_status, 50)
        self.pub_turn_command = self.create_publisher(Command, "/turn_robot/command", 10)
        self.pub_human_detection_command_scan = self.create_publisher(String, "/human_detection/command/scan", 10)
        self.pub_human_detection_command = self.create_publisher(String, "/human_detection/command", 10)

    def callback_command(self, msg: String):
        if not msg.data == "start":
            return
        time.sleep(1)
        # scanの開始
        self.pub_human_detection_command_scan.publish(String(data="xyz"))  # こいつだけメッセージが抜けることがある
        self.pub_human_detection_command_scan.publish(String(data="odometry"))
        self.pub_human_detection_command_scan.publish(String(data="image"))

        # 回転の開始
        self.pub_turn_command.publish(Command(command="START", content="360"))

    def callback_turn_status(self, msg: String):
        if not msg.data == "FINISH":
            return
        print("データ取得終了", flush=True)
        self.pub_human_detection_command_scan.publish(String(data="stop"))
        self.pub_human_detection_command.publish(String(data="sampling"))


def main():
    rclpy.init()
    node = HumanDetectionScanMain("HumanDetectionScanMain")
    rclpy.spin(node)


if __name__ == '__main__':
    main()
