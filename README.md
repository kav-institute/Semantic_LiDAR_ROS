# Semantic_LiDAR_ROS [![arXiv Badge](https://img.shields.io/badge/arXiv-B31B1B?logo=arxiv&logoColor=fff&style=flat)](https://arxiv.org/abs/2504.21602) [![ResearchGate Badge](https://img.shields.io/badge/ResearchGate-0CB?logo=researchgate&logoColor=fff&style=flat)](https://www.researchgate.net/publication/391328948_Real_Time_Semantic_Segmentation_of_High_Resolution_Automotive_LiDAR_Scans) [![Zenodo Badge](https://img.shields.io/badge/Zenodo-1682D4?logo=zenodo&logoColor=fff&style=flat)]([https://zenodo.org/records/14906179](https://doi.org/10.5281/zenodo.14677379)) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14677379.svg)](https://doi.org/10.5281/zenodo.14677379)

A docker container with ROS2 and a ROS2 Package for LiDAR semantic segmentation
![rgbImage](images/rviz_screenshot_2024_06_04-12_34_34.png)

## Environment
### Setup
Read the [DATA.md](dataset/DATA.md) to learn how to configure the demonstration system.

### Assets [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14677379.svg)](https://doi.org/10.5281/zenodo.14677379)
We created an asset pack containing meshes, sensor recordings and pretrained models.
You can download the assets from Zenodo. 
> [!CAUTION]
> The content of the asset pack is not provided by Apache 2.0 License.

[https://zenodo.org/records/14677379](https://zenodo.org/records/14677379)

### VS-Code
The project is designed to be delevoped within vs-code IDE using remote container development.

### Setup Docker Container
```bash
# Enable xhost in the terminal
sudo xhost +

# Add user to environment
sh setup.sh

# Build the image from scratch using Dockerfile, can be skipped if image already exists or is loaded from docker registry
docker compose build --no-cache

# Start the container
docker compose up -d

# Stop the container
docker compose down
```

> [!CAUTION]
> xhost + is not a save operation!

### Set Up in VS Code
#### Possibility 1 (Preferred)

#### 1. Step: Executing the first task "rviz start" to start rviz 
Hit menu item "Terminal" and chose "Run Task...":
 -> execute the task "rviz start" -> Rviz window opens 
 
#### 2. Step: Executing second task "SemanticLiDAR start" to start the Semantic Node: 
Hit menu item "Terminal" and chose "Run Task...":
 -> execute the task "SemanticLiDAR start" -> Rviz window shows the output of the semantic node

#### Possibility 2 
In VS Code open two terminals:
```bash
# Terminal 1, start RVIZ
appuser@yourpc:~/ros2_ws$ source /opt/ros/humble/setup.bash
appuser@yourpc:~/ros2_ws$ source install/local_setup.bash
appuser@yourpc:~/ros2_ws$ ros2 run rviz2 rviz2 -d semantic_lidar.rvizsemantic

# Terminal 2, start SemanticLiDAR Node
appuser@yourpc:~/ros2_ws$ source /opt/ros/humble/setup.bash
appuser@yourpc:~/ros2_ws$ source install/local_setup.bash
appuser@yourpc:~/ros2_ws$ colcon build
appuser@yourpc:~/ros2_ws$ ros2 run semantic_lidar_package semantic_lidar_node
```

### Reference System
```bash
OS: Ubuntu 22.04.4 LTS x86_64 
Host: B550 AORUS ELITE 
Kernel: 6.8.0-49-generic 
CPU: AMD Ryzen 9 3900X (24) @ 3.800G 
GPU: NVIDIA GeForce RTX 3090 
Memory: 32031MiB                      
```

### Cycle Time
```
[INFO] [1737031067.198220009]: Cycle Time Read Data: 3332684 nanoseconds, 3359665 nanoseconds
[INFO] [1737031067.212568632]: Cycle Time Inference: 13737361 nanoseconds 17725170 nanoseconds
[INFO] [1737031067.214456973]: Cycle Time Vis: 1351017 nanoseconds 19665629 nanoseconds
[INFO] [1737031067.229298125]: Cycle Time Publish Image: 14433504 nanoseconds 34605027 nanoseconds
[INFO] [1737031067.237311043]: Cycle Time Publish PC: 7383180 nanoseconds 42416655 nanoseconds
```

## Model Training
Check out the following repo if you want to learn how the models are trained:
[https://github.com/kav-institute/SemanticLiDAR](https://github.com/kav-institute/SemanticLiDAR)

<a name="license"></a>
## License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details. Note that the data, assets, and models are provided by a different licence!


## Citation:
```
@misc{reichert2025realtimesemanticsegmentation,
      title={Real Time Semantic Segmentation of High Resolution Automotive LiDAR Scans}, 
      author={Hannes Reichert and Benjamin Serfling and Elijah Schüssler and Kerim Turacan and Konrad Doll and Bernhard Sick},
      year={2025},
      eprint={2504.21602},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2504.21602}, 
}

@dataset{reichert_2025_14906179,
  author       = {Reichert, Hannes and
                  Schüssler, Elijah and
                  Serfling, Benjamin and
                  Turacan, Kerim and
                  Doll, Konrad and
                  Sick, Bernhard},
  title        = {SemanticTHAB: A High Resolution LiDAR Dataset},
  month        = feb,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.14906179},
  url          = {https://doi.org/10.5281/zenodo.14906179},
}
```


## Contributors
<a href="https://github.com/kav-institute/Semantic_LiDAR_ROS/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=kav-institute/Semantic_LiDAR_ROS" />
</a>

Made with [contrib.rocks](https://contrib.rocks).
