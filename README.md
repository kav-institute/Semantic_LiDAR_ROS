# Semantic_LiDAR_ROS

A docker container with ROS2 and a ROS2 Package for LiDAR semantic segmentation
![rgbImage](images/rviz_screenshot_2024_05_06-07_43_07.png)

## Development environment:

### VS-Code:
The project is designed to be delevoped within vs-code IDE using remote container development.

### Setup Docker Container
```bash
# Enable xhost in the terminal
sudo xhost +

# Add user to environment
sh setup.sh

# Build the image from scratch using Dockerfile, can be skipped if image already exists or is loaded from docker registry
docker-compose build --no-cache

# Start the container
docker-compose up -d

# Stop the container
docker-compose down
```

### Set Up in VS Code
#### Possibility 1: (Preferred)

#### 1. Step: Executing the first task "rviz start" to start rviz 
Hit menu item "Terminal" and chose "Run Task...":
 -> execute the task "rviz start" -> Rviz window opens 
 
#### 2. Step: Executing second task "SemanticLiDAR start" to start the Semantic Node: 
Hit menu item "Terminal" and chose "Run Task...":
 -> execute the task "SemanticLiDAR start" -> Rviz window shows the output of the semantic node



#### Possibility 2: 
In VS Code open two terminals:
```bash
# Terminal 1, start RVIZ
appuser@taurus:~/ros2_ws$ source /opt/ros/humble/setup.bash
appuser@taurus:~/ros2_ws$ source install/local_setup.bash
appuser@taurus:~/ros2_ws$ ros2 run rviz2 rviz2 -d semantic_lidar.rvizsemantic

# Terminal 2, start SemanticLiDAR Node
appuser@taurus:~/ros2_ws$ source /opt/ros/humble/setup.bash
appuser@taurus:~/ros2_ws$ source install/local_setup.bash
appuser@taurus:~/ros2_ws$ colcon build
appuser@taurus:~/ros2_ws$ ros2 run semantic_lidar_package semantic_lidar_node
```
### Model Zoo
You can download pre-trained models from our model zoo:
| Dataset | Backbone | Parameters | Inference Time¹ | mIoU² | Status 
|:-------:|:--------:|:----------:|:---------------:|:----:|:------:|
|SemanticKitti| [[KITTI_RN18]](https://drive.google.com/drive/folders/1blLMyAXlmSCHIvQhBRWdbkCvDqQtW4AR?usp=sharing) |  18 M      |  10ms  | 51.72%  | $${\color{green}Online}$$ 
|SemanticKitti| [[KITTI_RN34]](https://drive.google.com/drive/folders/1mDyPiZBHOi1mDpw-tvoqWRuKqjcod6N4?usp=sharing) |  28 M      |  14ms  | 57%  | $${\color{green}Online}$$ 
|SemanticTHAB³| [[THAB_RN18]](https://de.wikipedia.org/wiki/HTTP_404) |  18 M      |  10ms  | --  | $${\color{red}Offline}$$
|SemanticTHAB³| [[THAB_RN34]](https://drive.google.com/drive/folders/1tmyw1RNRtcm3tHld2owxVHm1-2Fvrnzn?usp=sharing) |  28 M      |  14ms  | 72%  | $${\color{green}Online}$$ 

For this demo we use the THAB_RN34.
Download the .pth file to:
```bash
├── dataset
│   ├── model_zoo
│   │   └── THAB_RN34
│   │   │   └── model_final.pth
```
