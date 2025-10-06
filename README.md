# MicroROS-Workspaces
## How to setup remote VS Code
Edit RPi5 files on VS Code remotely from your computer/laptop.

### 1. Download Remote SSH extension
- Open VS Code → Extensions tab
- Search for `Remote - SSH` → install the first result.  

### 2. Connect to Raspberry Pi from VS Code
1. Press F1 → type `Remote-SSH: Connect to Host...` → select your host. Add new host for first time: `ssh pi@ip-address`
2. VS Code will open a new window and will prompt for password. 
3. Once connected, you can open any folder on rpi5 to edit files. 

### 3. Clone Repo
- Clone repo at `/home/pi/`
```shell
git clone https://github.com/paprika-flow/microROS_rpi5_Workspaces
```
### 4. Run modified_ros2_humble.sh 
- Modified ros2_humble links workspace folder to the container that starts so any changes made to files inside the workspace stays saved on host files and will not be deleted when container is closed. 
- Any changes made in either place will show up in the other immediately
```shell
sh ~/microROS_rpi5_Workspaces/modified_ros2_humble.sh 
```

### 5. Using Dev Containers extension 
- To work in the container created by `modified_ros2_humble.sh` on vscode you can use the Dev Containers extension.
1. Press F1 → type `Dev Containers: Attach to Running Containers...` → select your container.
    - Make sure you don't connect to micro ros agent container if you ran `start_agent_rpi5.sh` only ros-humble container 

## To Do
- Create a database of images to test segmentaiton methods
- Raspberry PI on eduroam
- updtae documentation 
