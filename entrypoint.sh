#!/bin/bash
source /opt/ros/humble/setup.bash
cd /home/appuser/ros2_ws/src

# this is done to generate the skeleton for the ro node 
ros2 pkg create --build-type ament_python --license Apache-2.0 --node-name semantic_lidar_node semantic_lidar_package
cd ..
colcon build
#source install/local_setup.bash

#colcon --log-base /home/appuser/ros2_ws/tmp/log build --install-base /home/appuser/ros2_ws/tmp/install --#build-base /home/appuser/ros2_ws/tmp/build --packages-select object_tracker demo_launch #internal_interfaces diprolea_interfaces internal_services
#. /home/appuser/ros2_ws/tmp/install/setup.bash

#ros2 launch demo_launch object_tracker.launch.py

echo "--- Additional installations..."
pip install opencv-python
apt-get install libdynamicedt3d-dev
echo "--- Additional container setup completed, ready for work..."
tail -F /dev/null
