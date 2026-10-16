import os
from glob import glob


from setuptools import find_packages, setup

package_name = 'nova_carter_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
    (
        'share/ament_index/resource_index/packages',
        ['resource/' + package_name]
    ),

    (
        'share/' + package_name,
        ['package.xml']
    ),

    (
        os.path.join(
            'share',
            package_name,
            'launch'
        ),
        glob('launch/*.launch.py')
    ),
],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='youssef',
    maintainer_email='youssef@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'rgb_listener = nova_carter_perception.rgb_listener:main',
        'yolo_detector = nova_carter_perception.yolo_detector:main',
        'lidar_camera_projection = nova_carter_perception.lidar_camera_projection:main',
        'lidar_yolo_fusion = nova_carter_perception.lidar_yolo_fusion:main',
        'right_yolo_detector = nova_carter_perception.yolo_detector_right:main',
        'right_lidar_yolo_fusion = nova_carter_perception.right_lidar_yolo_fusion:main',
        'multi_camera_fusion = nova_carter_perception.multi_camera_fusion:main',
        'trajectory_controller = nova_carter_perception.trajectory_controller:main',
        'fusion_video_recorder = nova_carter_perception.fusion_video_recorder:main',
    ],
},

)
