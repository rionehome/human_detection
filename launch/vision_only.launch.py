from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import LogInfo


def generate_launch_description():
    return LaunchDescription([
        #############################################################
        LogInfo(
            msg="launch realsense_ros2_camera"
        ),
        Node(
            package="realsense_ros2_camera",
            node_executable="realsense_ros2_camera",
            output="screen"
        ),
        #############################################################
        LogInfo(
            msg="launch face_predictor"
        ),
        Node(
            package="face_predictor",
            node_executable="face_predictor",
            output="screen"
        ),
        #############################################################
        LogInfo(
            msg="launch human_detection"
        ),
        Node(
            package="human_detection",
            node_executable="scan_main",
            output="screen"
        ),
        Node(
            package="human_detection",
            node_executable="scan_image",
            output="screen"
        ),
        Node(
            package="human_detection",
            node_executable="scan_odometry",
            output="screen"
        ),
        Node(
            package="human_detection",
            node_executable="scan_xyz",
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
    ])
