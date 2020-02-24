from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="realsense_ros2_camera",
            node_executable="realsense_ros2_camera",
            output="screen"
        ),
        Node(
            package="human_detection",
            node_executable="scan",
            output="screen"
        ),
        Node(
            package="human_detection",
            node_executable="predict",
            output="screen"
        ),
        Node(
            package="human_detection",
            node_executable="calculation",
            output="screen"
        ),
        Node(
            package="face_predictor",
            node_executable="face_predictor",
            output="screen"
        ),
    ])
