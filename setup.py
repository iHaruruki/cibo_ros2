from glob import glob
import os
from setuptools import find_packages, setup

package_name = 'cibo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name, 'rviz'),   glob('rviz/*.rviz')),
        (os.path.join('share', package_name), glob('urdf/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Haruki Isono',
    maintainer_email='haruki.isono861@gmail.com',
    description='Eating Behavior Recognition for Elderly People.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'top_camera_node = cibo.top_camera:main',
            'top_camera_depth_node = cibo.top_camera_depth:main',
            'front_camera_node = cibo.front_camera:main',
            'front_camera_depth_node = cibo.front_camera_depth:main',
            'image_show_node = cibo.image_show:main',
            'compressed_image_show_node = cibo.compressed_image_show:main',
            'dual_camera_sync_node_node = cibo.dual_camera_sync:main',
            'check_image_compressed_node = cibo.check_image_compressed:main',
        ],
    },
)
