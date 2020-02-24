from setuptools import setup

package_name = 'human_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=[],
    py_modules=[
        'human_detection.scan',
        'human_detection.predict',
        'human_detection.calculation',
    ],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['launch/human_detection.launch.py']),
    ],
    install_requires=['setuptools', 'numpy', 'launch'],
    zip_safe=True,
    maintainer='hirorittsu',
    maintainer_email='zq6295migly@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'scan = human_detection.scan:main',
            'predict = human_detection.predict:main',
            'calculation = human_detection.calculation:main',
        ],
    },
)
