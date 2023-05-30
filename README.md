# DRL_Navigation_Robot_ROS2_Foxy
 Deep Reinforcement Learning for mobile robot navigation in ROS Gazebo 11 simulator. Using Twin Delayed Deep Deterministic Policy Gradient (TD3) neural network, a robot learns to navigate to a random goal point in a simulated environment while avoiding obstacles. Obstacles are detected by laser readings and a goal is given to the robot in polar coordinates. Trained in ROS Gazebo 11 simulator with PyTorch. Tested with ROS2 Foxy on Ubuntu 20.04 with python 3.8.10 and pytorch 1.10.

TD3 Network Implementation :

TD3 is an actor-critic type of network similar to DDPG. That means that there is an “actor” network that calculates an action to perform, and a “critic” network, that estimates, how good is this action. In a simple form, TD3 architecture is an extension of DDPG architecture to solve the problem of overestimating the Q-value. It does so by introducing a second critic network within the loop and selecting the output from the one that produces the lower Q-value estimations. (Once again, a mathematical and algorithmic background overview can be obtained here.) Therefore, we need to create an actor-network that will take the environmental state as input and output action for the robot to take. Also, we need to create two critic networks that will take the environmental state as well as the action from the actor-network as inputs and will output the estimated value of this state-action pair.

<p align="center">
    <img width=100% src="https://github.com/toxuandung/DRL_Navigation_Robot_ROS2_Foxy/assets/101309710/484631fb-669f-44b5-8c6d-b9e7d1db250e">
</p> 

Robot state :
- Distance to goal
- Theta
- Translational Velocity
- Angular Velocity
- Distance to an obstacle at each 9-degree interval within a 180-degree range in front of a robot from LIDAR (Light Detection and Ranging) sensor

Reward :

r = v - |ω|

Training environment :

<p align="center">
    <img width=50% src="https://github.com/toxuandung/DRL_Navigation_Robot_ROS2_Foxy/blob/main/Training_env.png">
</p>

## Installation
Main dependencies: 

* [ROS2 Foxy](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html)
* [PyTorch](https://pytorch.org/get-started/locally/)
* [Tensorboard](https://github.com/tensorflow/tensorboard)
