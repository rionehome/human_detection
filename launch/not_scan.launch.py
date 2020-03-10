from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import LogInfo


def generate_launch_description():
    return LaunchDescription([
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
            msg="launch gender_predictor"
        ),
        Node(
            package="gender_predictor",
            node_executable="gender_predictor",
            output="screen"
        ),
        #############################################################
        LogInfo(
            msg="launch human_detection"
        ),
        Node(
            package="human_detection",
            node_executable="sampling",
            output="screen"
        ),
        Node(
            package="human_detection",
            node_executable="calculation",
            output="screen"
        ),
        Node(
            package="human_detection",
            node_executable="labeling",
            output="screen"
        ),
    ])
