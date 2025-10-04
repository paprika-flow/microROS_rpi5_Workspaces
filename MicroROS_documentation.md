# MicroROS Domcumentation
## start_agent_rpi5.sh
This is a shell script that starts the mirco ROS agent on the raspberry pi and allows communication to the esp32's micro ros client  through serial (USB C). Micro ros client on esp32 sends messages to the agent on the rpi5 allowing for subscribing or publishing topics. 

## ros2_humble.sh
This is a shell script that runs a docker command to start ros2 humble (humble is a version of ros2) comes with the robot. 

Allows interactivity with terminal, with access to raspberry pi hardware, network ,dsiplay, inputs, camera, and removes most security restrictions. 


## How to setup remote VS Code
Edit RPi5 files on VS Code remotely from your computer/laptop.

### 1. Download Remote SSH extension
- Open VS Code → Extensions tab
- Search for `Remote - SSH` → install the first result.  


### 2. Download Dev Containers extension 
- To work in the container created by `ros2_humble.sh` you can use this extension.
    - Mount workspace folder 



### 3. Connect to Raspberry Pi from VS Code
1. Press F1 → type `Remote-SSH: Connect to Host...` → select your host.  
2. VS Code will open a new window connected to the Pi.  
3. Open your project folder via File → Open Folder on the Pi.  


