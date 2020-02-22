from setuptools import setup

package_name = 'human_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=[],
    py_modules=[
        'script.human_detection_scan',
        'script.human_detection_predict',
        'script.human_detection_calculation',
    ],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hirorittsu',
    maintainer_email='zq6295migly@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'scan = script.human_detection_scan:main',
            'predict = script.human_detection_predict:main',
            'calculation = script.human_detection_calculation:main',
        ],
    },
)
