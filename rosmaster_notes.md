## TO DO 
- [x] ROS2 install
- [x] Get dustnv to work
- [ ] Try to get the yahboom drivers into dustynv

## Restarting At ROS1 + Flashing ROS2
-  When i started it was on ros1 from yahboom
    - Was able to get the ir camera to work by simply plugging the usb directly into to the nano.
-  Decided to go to to ROS2 because it has more support and is more modern.
-  Need to reflash with yahbooms image.
    1.  First Download the [system file](https://drive.google.com/drive/folders/1_97RdpSyTEGgvsna6JX5hpxt4LjiztBV) for nano.
    2. next remove the 64GB usb from nano then plug into a compute (preferably linux so u can just simply use dd)
    3. See if the usb is mounted and if so unmount. 
        - `lsblk` to list block devices. find the matching size block (most likely /dev/sdb)
        - `sudo unmount /dev/sdX*` where x is the block name
    4. now use dd to copy byte by byte of the `.img` or `.iso` file to the usb :
    
    ```bash
    sudo dd if=/path/to/your/image.iso of=/dev/sdX bs=4M status=progress conv=fsync
    ```
    5. simply eject with `sudo eject /dev/sdX` then plug in the usb to the jetson nano.
    6. It should boot directly to the desktop on a gui or if using ssh just try `ssh jetson@192.168.1.11` (first connect to `ROSMASTER` wifi network pw is usually `yahboom` or `1234568`) to make sure it is sshable look at the small screen on the top it should contain the ip if any.
    7. Make sure there is a file called `run_docker.sh`
    8. **Make sure to update apt**
    ```bash
    sudo apt-get update
    sudo apt-get upgrade
    ```
---

## Trying to get Dusty - NV containers working. 

#### WHY? 
Because the default yahboom containers are not jetson nano friendly. 
- [Dusty NV Containers](https://github.com/dusty-nv/jetson-containers) are containers maintained by the community and make working with the gpu easier on docker containers with ros2 already enabled
- We are not using the default yahboom containers because they dont already support using the gpu in the ros2 container (it is too difficult to alter the container to support the gpu)

#### Steps taken:
1. First clone the [dusty-nv containers](https://github.com/dusty-nv/jetson-containers) github. 
```bash
git clone https://github.com/dusty-nv/jetson-containers
cd jetson-containers/
```
1. install the containers with `bash install.sh` 
   - If any issues arise, go to an older version with git. I used `git checkout 5645241` to go to a version without black code formatter
2. Edit `/etc/docker/daemon.json` - Be careful doing this if that file already has things in there. if so append this to the end of the json structure (make sure to keep json formatting correct) : 
```bash

{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
```

4. run `sudo systemctl restart docker` to restart docker to make changes
5. To run the container use :
   ```bash
    ~/jetson-containers/run.sh --device /dev/myserial --device /dev/i2c-1 dustynv/ros:humble-pytorch-l4t-r32.7.1
    ```
    - This command will pull the container from docker the first time it is run.
    - we are using the container that comes with pytorch so we can use the segmentation models.
    - to make sure the gpu is working run: 
  ```bash
  python3 -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
  ```
   - If this prints true the gpu has been passed to the container! If not good luck. 
---
## Getting the Yahboom drivers
We are on a different container so we need the yahboom drivers to be able to talk to the Rosmaster hardware. 
- Here are the directions to install the drivers from [Yahboom Repo ](https://github.com/YahboomTechnology/ROSMASTERX3/blob/37f87909d3744d2ca579066b90e65ae54deea1c8/04.X3-ROS2-Tutorials/05.ROSMASTER%20Basic%20course/3.%20Install%20the%20Rosmaster%20driver%20library.pdf)
- The instructions are somewhat hard to understand so here are my instructions:

#### Steps taken:
1. Enter the yahboom container with `sh ~/run_docker.sh`
2. navigate to `~/yahboomcar_ros2_ws/software/py_install_V3.3.1` to validate that it exists. 
3. Open a new terminal session while the container is running. Find the running container ID with `docker ps`Then copy the driver library folder to the host with this command: 
```bash 
docker cp [container ID]:/root/yahboomcar_ros2_ws/software/py_install_V3.3.1 ~/Desktop/
```
if an error arises when doing this command like `not a directory` then use this command:
```bash
#This command compresses the folder then sends it to the host and automatically uncompresses the folder.
mkdir -p ~/yahboom_drivers && docker exec c3f1c84f0f75 tar -C /root/yahboomcar_ros2_ws/software/ -cf - py_install_V3.3.1 | tar -C ~/yahboom_drivers -xf -
```
4. Now that the driver library is on the host we have to move it to the dusty nv container. We can write a docker file to build a new image with the dusty nv image as a base and simply build the drivers inside. 
```bash 
cd yahboom_drivers
touch Dockerfile
```