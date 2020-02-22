import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2
import numpy as np


class HumanDetectionScan(Node):

    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.is_start = False
        self.color_image = None
        self.stack_around_info = []
        self.create_subscription(String, "/human_detection/command", self.callback_command, 10)
        self.create_subscription(Image, "/camera/color/image_raw", self.callback_color_image, 10)
        self.create_subscription(PointCloud2, "/camera/aligned_depth_to_color/color/points", self.callback_point_cloud,
                                 1)

    def callback_command(self, msg):
        if msg.data == "start":
            self.is_start = True
        else:
            self.is_start = False
            print("not start.")

    def callback_color_image(self, msg: Image):
        # self.color_image = np.asarray(msg.data, dtype=np.uint8).view(dtype=np.uint16).reshape((msg.height, msg.width))
        self.color_image = np.asarray(msg.data).reshape((msg.height, msg.width, 3))

    def callback_point_cloud(self, msg: PointCloud2):
        real_data = np.asarray(msg.data, dtype=np.uint8).view(dtype=np.float32).reshape((msg.height, msg.width, 8))

        x = real_data[:, :, 0]
        y = real_data[:, :, 1]
        z = real_data[:, :, 2]

        if self.color_image is None:
            return

        distance_map = np.sqrt(x * x + y * y + z * z)
        distance_image = (z * 25).astype(int).astype(np.uint8)
        cv2.imshow("depth", distance_image)
        cv2.imshow("color", self.color_image)
        blended = cv2.addWeighted(src1=cv2.cvtColor(self.color_image, cv2.COLOR_RGB2BGR), alpha=0.7,
                                  src2=cv2.cvtColor(distance_image, cv2.COLOR_GRAY2RGB), beta=0.3, gamma=0)
        cv2.imshow("blended", blended)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = HumanDetectionScan("HumanDetectionScan")
    rclpy.spin(node)


if __name__ == '__main__':
    main()
