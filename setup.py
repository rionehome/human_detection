import os

from setuptools import setup

package_name = 'human_detection'


def create_install_files(paths: list):
    target_data_files = [('share/ament_index/resource_index/packages', ['resource/' + package_name]),
                         ('share/' + package_name, ['package.xml'])]

    for path in paths:
        for root, dirs, files in os.walk(path):
            print(root)
            target_data_files.append(('lib/{}/{}'.format(package_name, root), []))
            for file in files:
                print('{}/{}'.format(root, file))
                target_data_files[-1][1].append('{}/{}'.format(root, file))
    return target_data_files


# print(create_install_files(['human_detection/lib']))

setup(
    name=package_name,
    version='0.0.0',
    packages=[],
    py_modules=[
        'human_detection.scan_main',
        'human_detection.scan_image',
        'human_detection.scan_odometry',
        'human_detection.scan_xyz',
        'human_detection.predict',
        'human_detection.calculation',
    ],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['launch/human_detection.launch.py']),
        ('share/' + package_name, ['launch/vision_only.launch.py']),
        ('lib/human_detection/lib',
         ['human_detection/lib/__init__.py',
          'human_detection/lib/module.py',
          'human_detection/lib/Logger.py'])
    ],
    install_requires=['setuptools', 'numpy', 'launch', 'joblib', 'opencv-python', 'matplotlib', 'sklearn', 'pandas'],
    zip_safe=True,
    maintainer='hirorittsu',
    maintainer_email='zq6295migly@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'scan_main = human_detection.scan_main:main',
            'scan_image = human_detection.scan_image:main',
            'scan_odometry = human_detection.scan_odometry:main',
            'scan_xyz = human_detection.scan_xyz:main',
            'predict = human_detection.predict:main',
            'calculation = human_detection.calculation:main',
        ],
    },
)
