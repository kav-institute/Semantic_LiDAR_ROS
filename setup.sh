#!/bin/bash

# Get the current user's name and ID

#$(whoami)
user_id=$(id -u)
user_name=appuser

# Write the user name and ID to the .env file
echo "USER_ID=$user_id" > .env
echo "USER_NAME=$user_name" >> .env
echo "ROS_DISTRO=humble" >> .env

echo "Environment variables written to .env file:"
echo "USER_NAME=$user_name"
echo "USER_ID=$user_id"
