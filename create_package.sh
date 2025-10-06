#!/bin/bash

read -p "Enter package name: " pkg_name

full_pkg="pkg_${pkg_name}_py"

echo "Creating ROS 2 package: ${full_pkg}"
ros2 pkg create "${full_pkg}" \
  --build-type ament_python \
  --dependencies rclpy \
  --node-name "${pkg_name}"

echo "Package '${full_pkg}' created successfully."
