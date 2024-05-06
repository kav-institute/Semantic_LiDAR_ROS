from setuptools import find_packages, setup

package_name = 'semantic_lidar_package'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hannes.reichert',
    maintainer_email='hannes.reichert@th-ab.de',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'semantic_lidar_node = semantic_lidar_package.semantic_lidar_node:main'
        ],
    },
)
