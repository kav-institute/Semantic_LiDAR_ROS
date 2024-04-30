# Semantic_LiDAR_ROS

A docker container with ROS2 and a ROS2 Package for LiDAR semantic segmentation

[![Watch the video](https://cdn.discordapp.com/attachments/709432890458374204/1219546130115727390/image.png?ex=66309bd7&is=661e26d7&hm=c48cbefebdc49abcba54b0350bd200d4fae5accf0a629c695a429e82c0eac7f9&)](https://drive.google.com/file/d/1R7l4302yjyHZzcCP7Cm9vKr7sSnPDih_/view)
## Development environment:

### VS-Code:
The project is designed to be delevoped within vs-code IDE using remote container development.

### Setup Docker Container
In docker-compse.yaml all parameters are defined.
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
docker compose down
```

###
You can download pre-trained models from our model zoo:

| Dataset | Backbone | Parameters | Inference Time¹ | mIoU² | Status 
|:-------:|:--------:|:----------:|:---------------:|:----:|:------:|
|SemanticKitti| [[THAB_RN18]](https://drive.google.com/drive/folders/1blLMyAXlmSCHIvQhBRWdbkCvDqQtW4AR?usp=sharing) |  18 M      |  10ms  | 51.72%  | $${\color{green}Online}$$ 
|SemanticKitti| [[THAB_RN34]](https://drive.google.com/drive/folders/1mDyPiZBHOi1mDpw-tvoqWRuKqjcod6N4?usp=sharing) |  28 M      |  14ms  | 57%  | $${\color{green}Online}$$ 
|SemanticTHAB³| [[THAB_RN18]](https://de.wikipedia.org/wiki/HTTP_404) |  18 M      |  10ms  | --  | $${\color{red}Offline}$$
|SemanticTHAB³| [[THAB_RN34]](https://drive.google.com/drive/folders/1tmyw1RNRtcm3tHld2owxVHm1-2Fvrnzn?usp=sharing) |  28 M      |  14ms  | 72%  | $${\color{green}Online}$$ 
