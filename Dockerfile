################################## ros-core ##################################
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
  ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
  apt-get update && \
  apt-get install -q -y --no-install-recommends tzdata && \
  rm -rf /var/lib/apt/lists/*

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
  dirmngr \
  gnupg2 \
  && rm -rf /var/lib/apt/lists/*

# setup sources.list
RUN echo "deb http://packages.ros.org/ros2/ubuntu jammy main" > /etc/apt/sources.list.d/ros2-latest.list

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV ROS_DISTRO humble

# install ros2 packages
RUN apt-get update && apt-get install -y --no-install-recommends \
  ros-humble-ros-core=0.10.0-1* \
  && rm -rf /var/lib/apt/lists/*

################################## end of ros-core ##################################

################################## ros-base ##################################

# taken from official ros base at https://github.com/osrf/docker_images/blob/master/ros/humble/ubuntu/jammy/ros-base/Dockerfile

# This is an auto generated Dockerfile for ros:ros-base
# generated from docker_images_ros2/create_ros_image.Dockerfile.em

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
  build-essential \
  git \
  python3-colcon-common-extensions \
  python3-colcon-mixin \
  python3-rosdep \
  python3-vcstool \
  && rm -rf /var/lib/apt/lists/*

# bootstrap rosdep
RUN rosdep init && \
  rosdep update --rosdistro $ROS_DISTRO

# setup colcon mixin and metadata
RUN colcon mixin add default \
  https://raw.githubusercontent.com/colcon/colcon-mixin-repository/master/index.yaml && \
  colcon mixin update && \
  colcon metadata add default \
  https://raw.githubusercontent.com/colcon/colcon-metadata-repository/master/index.yaml && \
  colcon metadata update

# install ros2 packages
RUN apt-get update && apt-get install -y --no-install-recommends \
  ros-humble-ros-base=0.10.0-1* \
  && rm -rf /var/lib/apt/lists/*

################################## end of ros-base ##################################

#basic installs
# if needed add openssh-server
# restart openssh-server on container start
RUN apt-get update && apt-get install -y \
  python3-opencv ca-certificates python3-dev git wget sudo ninja-build
RUN ln -sv /usr/bin/python3 /usr/bin/python

# create a non-root user
ARG USER_ID=1001

RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

# python things
ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
  python3 get-pip.py --user && \
  rm get-pip.py

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install --user tensorboard cmake onnx   # cmake from apt-get is too old


RUN pip install --user torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121



RUN sudo apt-get update -qq \
  && sudo apt-get -y install ros-humble-rviz2 \
  && sudo apt-get autoclean && sudo apt-get clean && sudo apt-get -y autoremove \
  && sudo rm -rf /var/lib/apt/lists/*


ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,compat32,utility

RUN sudo apt-get update && sudo apt-get install -y \
  llvm \
  freeglut3 \ 
  freeglut3-dev \
  tar \
  libcanberra-gtk-module \
  libcanberra-gtk3-module \
  && sudo rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/appuser/.vscode
RUN mkdir -p /home/appuser/ros2_ws/tmp
RUN mkdir -p /home/appuser/ros2_ws/src && cd /home/appuser/ros2_ws/src
WORKDIR /home/appuser/ros2_ws/src

# install neccessary deps (e.g req file)
COPY requirements.txt /home/appuser
RUN pip install --no-cache-dir --user -r /home/appuser/requirements.txt

# ENV PYTHONPATH "${PYTHONPATH}:/home/appuser/ros2_ws/src/demo_utils"

# ADD CMAKE COMPILER
RUN sudo apt-get update && sudo apt-get -y install g++


COPY entrypoint.sh /home/appuser/entrypoint.sh
RUN sudo chmod +x /home/appuser/entrypoint.sh

ENTRYPOINT [ "bash" ]
RUN sudo chmod +x /home/appuser/entrypoint.sh

ENTRYPOINT [ "bash" ]
