[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: ./images/plot_of_rewards.png "plot of rewards"

# Collaboration and Competition Project Report

## 1. Introduction

For this project, we implement a Multi Agent DDPG to solve the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

## 2. Learning Algorithm

In order to solve the environment we implemented a MADDG model according to the [Multi-Agent Actor-Critic for Mixed
Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf) paper.

The MADDPG model we implemented has two identical DDPG agents having a shared replay buffer. The idea behid MADDPG is that each agent's critic network has access to the states and actions of the other agents at training, while at evalutation we do not need the critic network, hence, it can perform very well without having to know the other agents' states or actions at inference time.

In order to train the model, for each time step, we train the local networks using a batch of 256 (sampled experiences from a shared buffer of size 100 000) and update the target networks using soft updates.

The MADDPG and DDPG algorithms are implemented in [maddpg_agent.py](maddpg_agent.py).

## 3. Model Architecture

Each DDPG agent has four networks defined in [model.py](model.py):

- Local actor

- target actor

- local critic

- target critic

Where the local and target networks are identical.

The local network has two hidden fully connected layers (with relu activations) of size 400 and 300, and two outputs (with tanh activations).

While the critic network takes as input the states of all agents, has two hidden fully connected layers (with relu activations) of size 400 and 300, and concatenates the actions of all agents with the first hidden layer. All the activations of the critic are relu activations except for the output which is of size 1 and have no activation function.

We also added Ornstein-Uhlenbeck Noise to actions in the training phase to improve performances.

## 4. Hyperparameters Tuning

The hyperparameters where chosen using trial and error.

We used a :

- Learning rate of 1 e-4 and 1 e-3 for the actor and  the critic respectively.

- Buffer size of 100 000 for the experience replay

- Update rate of 1 (Update the agent every time step)

- Batch size of 256 for training

- Dicount factor Gamma of 0.99

- Tau coefficient of 5 e-3 for soft update of target parameters

- Initial noise value for epsilon = 3.0

- Noise decay value of 0.995

- Final noise value for epsilon = 0.0

## 5. Results

The agent was able to solve the environment after 3625 episodes with an average score of 0.51 for the last 100 episodes.

![plot of rewards][image2]

## 6. Ideas for Future Work

Even though the results where satisfying, we still can improve them by tuning the hyperparameters or by implementing other multi agents model like OpenAi 5 or Alpha Star from DeepMind and benchmark the results.
