from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import LogInfo


def generate_launch_description():
    return LaunchDescription([
        #############################################################
        LogInfo(
            msg="launch turn_robot"
        ),
        Node(
            package="turn_robot",
            node_executable="turn_robot",
        ),
        #############################################################
        LogInfo(
            msg="launch turtlebot_bringup"
        ),
        Node(
            package="ydlidar",
            node_executable="ydlidar_node",
            output="screen",
            parameters=["ydlidar.yaml"]
        ),
        Node(
            package="turtlebot_bringup",
            node_executable="turtlebot2",
            output="screen",
            parameters=["turtlebot2.yaml"]
        ),
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
