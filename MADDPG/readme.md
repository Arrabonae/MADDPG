# Multi-agent Deep Deterministic Policy Gradient (MADDPG)


**Episode 0**          |  **Episode 20000**
:--------------------:|:--------------------:


## Pettingzoo MPE environment for multi-agent RL
This repository is an example of Multi-agent Deep Deterministic Policy Gradient network architecture with centralised Critic and decentralised Actors. This implementation is fine-tuned for cooperative environment: **Simple Spread**. <br/>


## The environment - Simple Spread
The environment is a cooperative environment, where the goal is to move the agents to the goal locations. The agents are rewarded for being close to the goal locations. The agents are penalized for colliding with each other. The environment is considered solved when the agents get an average reward of +0.5 over 100 consecutive episodes. <br/>
More about the environment: https://pettingzoo.farama.org/environments/mpe/simple_spread/ <br/>

## Environment details
Action space: Continuous (low=0,high=1)(5)=[no_action, move_left, move_right, move_down, move_up] <br/>
Observation space: [self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, communication] <br/>

## Multi-agent Deep Deterministic Policy Gradient
MADDPG is a modified version of DDPG algorithm, in this case, we have multiple agents. The main difference between DDPG and MADDPG is that in DDPG we have an Actor-Critic network pair, while in MADDPG we have a n number of Actor networks (n is the number if agents in the game) and a single Critic network that takes as input the states of all agents and the actions of all agents. <br/>
The enviroment is designed in a way that each agent has its own observation space and action space which are not identical, however the Agents have a common (global) goal to achive therefore they must work together in order to maximise their individual rewards. <br/>
More about the MADDPG algorithm: https://arxiv.org/pdf/1706.02275.pdf<br/>


## Network architecture
![](plots/network.pdf)<br/>

## Results and benchmarks
The network was trained for 20000 episodes, and the results are shown below. <br/>
![](plots/MADDPG.png)<br/>

The results are benchmarked to DDPG algorithms on the same environment, same hyperparemeter setup and same number of episodes. 
<br/>
![](plots/MADDPG_DDPG.png)<br/>


## Notable Hyperparameters
- Learning and weight transfer every 100 episodes
- Episodes training: 20000
- Batch size: 1024
- Learning rate (Actor): 0.001
- Learning rate (Critic): 0.002
- Discount factor (Gamma): 0.95
- weight transfer (Tau): 0.01
- Memory Size: 10**6
- Noise: 0.01

I used RELU activation functions with sigmoid activation function for the output layer of the Actor network. The Critic network uses RELU activation functions with no activation function for the output layer. <br/>

## Prerequisites
* Pettingzoo 
* Numpy
* Tensorflow
* Matplotlib

This network was trained in Google Colab environment used GPU (Cuda), the code is omptimalised for CUDA environment. 

## Links to research papers and other repositories
Based on OpenAi's research paper: https://arxiv.org/pdf/1706.02275.pdf <br/>
