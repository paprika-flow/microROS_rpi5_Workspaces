# MicroROS Domcumentation
## start_agent_rpi5.sh
This is a shell script that starts the mirco ROS agent on the raspberry pi and allows communication to the esp32's micro ros client  through serial (USB C). Micro ros client on esp32 sends messages to the agent on the rpi5 allowing for subscribing or publishing topics. 

## ros2_humble.sh
This is a shell script that runs a docker command to start ros2 humble (humble is a version of ros2) comes with the robot. 

Allows interactivity with terminal, with access to raspberry pi hardware, network ,dsiplay, inputs, camera, and removes most security restrictions. 

