from setuptools import setup

package_name = 'human_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=[],
    py_modules=[
        'scripts.human_detection_scan',
    ],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'cv2', 'numpy'],
    zip_safe=True,
    maintainer='hirorittsu',
    maintainer_email='zq6295migly@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'human_detection_scan_node = scripts.human_detection_scan:main'
        ],
    },
)
