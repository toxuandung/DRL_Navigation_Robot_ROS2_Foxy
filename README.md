# DRL_Navigation_Robot_ROS2_Foxy
 Deep Reinforcement Learning for mobile robot navigation in ROS Gazebo 11 simulator. Using Twin Delayed Deep Deterministic Policy Gradient (TD3) neural network, a robot learns to navigate to a random goal point in a simulated environment while avoiding obstacles. Obstacles are detected by LIDAR (Light Detection and Ranging) sensor and a goal is given to the robot in polar coordinates. Trained in ROS Gazebo 11 simulator with PyTorch. Tested with ROS2 Foxy on Ubuntu 20.04 with python 3.8.10 and pytorch 1.10.
 <p align="center">
    <img width=70% src="https://github.com/toxuandung/DRL_Navigation_Robot_ROS2_Foxy/blob/main/Test_example_env1.gif">
</p>
TD3 Network Implementation :

TD3 is an actor-critic type of network similar to DDPG. That means that there is an “actor” network that calculates an action to perform, and a “critic” network, that estimates, how good is this action. In a simple form, TD3 architecture is an extension of DDPG architecture to solve the problem of overestimating the Q-value. It does so by introducing a second critic network within the loop and selecting the output from the one that produces the lower Q-value estimations. (Once again, a mathematical and algorithmic background overview can be obtained here.) Therefore, we need to create an actor-network that will take the environmental state as input and output action for the robot to take. Also, we need to create two critic networks that will take the environmental state as well as the action from the actor-network as inputs and will output the estimated value of this state-action pair.

```python
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2_s = nn.Linear(800, 600)
        self.layer_2_a = nn.Linear(action_dim, 600)
        self.layer_3 = nn.Linear(600, 1)

        self.layer_4 = nn.Linear(state_dim, 800)
        self.layer_5_s = nn.Linear(800, 600)
        self.layer_5_a = nn.Linear(action_dim, 600)
        self.layer_6 = nn.Linear(600, 1)

    def forward(self, s, a):
        s1 = F.relu(self.layer_1(s))
        self.layer_2_s(s1)
        self.layer_2_a(a)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(a, self.layer_2_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        s2 = F.relu(self.layer_4(s))
        self.layer_5_s(s2)
        self.layer_5_a(a)
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(a, self.layer_5_a.weight.data.t())
        s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        return q1, q2

# td3 network
class td3(object):
    def __init__(self, state_dim, action_dim, max_action):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        # Initialize the Critic networks
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action
        self.writer = SummaryWriter(log_dir="./src/td3/runs/train/tensorboard")
        # os.path.dirname(os.path.realpath(__file__)) + "/runs"
        self.iter_count = 0

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
```
<!-- <p align="center">
    <img width=100% src="https://github.com/toxuandung/DRL_Navigation_Robot_ROS2_Foxy/assets/101309710/484631fb-669f-44b5-8c6d-b9e7d1db250e">
</p> 

<p align="center">
    <img width=50% src="https://github.com/toxuandung/DRL_Navigation_Robot_ROS2_Foxy/blob/main/Actor.png">
</p> 

<p align="center">
    <img width=60% src="https://github.com/toxuandung/DRL_Navigation_Robot_ROS2_Foxy/blob/main/Critic.png">
</p> 

<p align="center">
    <img width=90% src="https://github.com/toxuandung/DRL_Navigation_Robot_ROS2_Foxy/blob/main/Td3.png">
</p>  -->

The detail of the network can be found in src/td3/scripts

The Robot and The Evironment :

We are trying to "find the optimal sequence of actions that lead the robot to a given goal”. There are two things to consider — the action and the environment that the action reacts to. In a mobile robot setting, it is quite easy to express the action in a mathematical form. It is the force applied to each actuator for the controllable degree of freedom. To put it simply, it is how much we want to move in any controllable direction.

    a = (v, ω)
    s = (laser_state + distance_to_goal + theta + previous_action)
    
- a is tuple action , v is translational velocity, ω is angular velocity
- s is state, laser_state are distances to an obstacle at each 9-degree interval within a 180-degree range in front of a robot from LIDAR sensor, theta is angles between the robot heading and the heading towards the goal

Reward :

if robot_reach_the_goal:

    r = 100
elif collision:

    r = -100.0
else:

    r = v - |ω| - r3  // r3 = (1 - smallest distance of robot to obstacles) if that distance < 1m else r3 = 0
    
r is the reward for each time step,
The idea behind it is that the robot needs to realize that it should be moving around and not just sitting in a single spot. By setting a positive reward for linear motion robot first learns that moving forward is good and rotating is not.Additionally, we add the term r3 which is calculated by our lambda function. This gives an additional negative reward if the robot is closer to any obstacle than 1 meter.

Training environment :

<p align="center">
    <img width=50% src="https://github.com/toxuandung/DRL_Navigation_Robot_ROS2_Foxy/blob/main/Training_env.png">
</p>

## Installation
Main dependencies: 

* [PyTorch](https://pytorch.org/get-started/locally/)
* [Tensorboard](https://github.com/tensorflow/tensorboard)
* [ROS2 Foxy](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html)

```shell
$ sudo apt install python3-colcon-common-extensions
$ sudo apt install ros-foxy-gazebo-ros-pkgs
$ sudo apt install ros-foxy-xacro
```
Clone the repository :

```shell
$ git clone https://github.com/toxuandung/DRL_Navigation_Robot_ROS2_Foxy.git
$ cd DRL_Navigation_Robot_ROS2_Foxy
```
Compile the workspace:

```shell
$ source /opt/ros/foxy/setup.bash
$ colcon build
$ source install/setup.bash
```
Training :
```shell
$ ros2 launch td3 training_simulation.launch.py
```
monitor the training process by tensorboard. Open the new terminal:

```shell
$ tensorboard dev upload --logdir     './src/td3/runs/train/tensorboard'
```
<p align="center">
    <img width=70% src="https://github.com/toxuandung/DRL_Navigation_Robot_ROS2_Foxy/blob/main/Tensorboard.PNG">
</p>

Training example :

<p align="center">
    <img width=70% src="https://github.com/toxuandung/DRL_Navigation_Robot_ROS2_Foxy/blob/main/Training_example.gif">
</p>

Testing :
```shell
$ ros2 launch td3 test_simulation.launch.py
```
Test example :

<p align="center">
    <img width=70% src="https://github.com/toxuandung/DRL_Navigation_Robot_ROS2_Foxy/blob/main/Test_example_env1.gif">
</p>

