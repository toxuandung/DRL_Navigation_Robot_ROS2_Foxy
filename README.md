# DRL_Navigation_Robot_ROS2_Foxy
 Deep Reinforcement Learning for mobile robot navigation in ROS Gazebo 11 simulator. Using Twin Delayed Deep Deterministic Policy Gradient (TD3) neural network, a robot learns to navigate to a random goal point in a simulated environment while avoiding obstacles. Obstacles are detected by laser readings and a goal is given to the robot in polar coordinates. Trained in ROS Gazebo 11 simulator with PyTorch. Tested with ROS2 Foxy on Ubuntu 20.04 with python 3.8.10 and pytorch 1.10.

Net work structure :
<p align="center">
    <img width=100% src="https://github.com/toxuandung/DRL_Navigation_Robot_ROS2_Foxy/assets/101309710/484631fb-669f-44b5-8c6d-b9e7d1db250e">
</p> 

Robot state :
- Distance to goal
- Theta
- Translational Velocity
- Angular Velocity
- Distance to an obstacle at each 9-degree interval within a 180-degree range in front of a robot from LIDAR (Light Detection and Ranging) sensor


Training environment :

<p align="center">
    <img width=50% src="https://github.com/toxuandung/DRL_Navigation_Robot_ROS2_Foxy/blob/main/Training_env.png">
</p>

## Installation
Main dependencies: 

* [ROS2 Foxy](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html)
* [PyTorch](https://pytorch.org/get-started/locally/)
* [Tensorboard](https://github.com/tensorflow/tensorboard)
