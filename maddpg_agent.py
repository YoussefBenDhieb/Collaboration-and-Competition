import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import DDPGModel
from OUNoise import OUNoise

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 5e-3  # for soft update of target parameters
LR_ACTOR = 1e-4  # learning rate for the actor
LR_CRITIC = 1e-3  # learning rate for the actor
UPDATE_EVERY = 1  # how often to update the network
EPS_START = 3.0  # first value of epsilon for noise to start with
EPS_END = 0.0  # last value of epsilon
EPS_DECAY = 0.995  # decay for epsilon

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MADDPG:
    """Multi Agents DDPG and shared replay buffer."""

    def __init__(
        self, state_size, action_size, seed=0, load_file=None, n_agents=2
    ):
        """
        Params
        ======
            action_size (int): dimension of each action
            seed (int): Random seed
            load_file (str): path of checkpoint file to load
            n_agents (int): number of distinct agents
            noise_start (float): initial noise weighting factor
            noise_decay (float): noise decay rate
            evaluation_only (bool): set to True to disable updating 
                                    gradients and adding noise
        """

        self.n_agents = n_agents
        self.agents_indexes = np.arange(self.n_agents)
        self.state_size = state_size
        self.action_size = action_size
        # create two agents, each with their own actor and critic
        self.agents = [
            DDPG_Agent(
                i, self.state_size, self.action_size, self.n_agents, seed
            )
            for i in range(self.n_agents)
        ]

    def step(self, states, actions, rewards, next_states, dones):
        experience = zip(
            self.agents, states, actions, rewards, next_states, dones
        )
        for i, e in enumerate(experience):
            agent, state, action, reward, next_state, done = e
            na_filtered = self.agents_indexes[self.agents_indexes != i]
            others_states = states[na_filtered]
            others_actions = actions[na_filtered]
            others_next_states = next_states[na_filtered]
            agent.step(
                state,
                action,
                reward,
                next_state,
                done,
                others_states,
                others_actions,
                others_next_states,
            )

    def act(self, states, add_noise=True):
        # pass each agent's state from the environment and calculate 
        # it's action
        all_actions = np.zeros(
            (self.n_agents, self.action_size), dtype=np.float
        )
        for agent in self.agents:
            all_actions[agent.agent_id, :] = agent.act(
                states[agent.agent_id], add_noise=add_noise
            )
        return all_actions  # reshape 2x2 into 1x4 dim vector

    def save(self, path="./saved_agents/"):
        for i, agent in enumerate(self.agents):
            torch.save(
                agent.actor_local.state_dict(),
                path + "actor_local_" + str(i) + ".pth",
            )
            torch.save(
                agent.critic_local.state_dict(),
                path + "critic_local_" + str(i) + ".pth",
            )
            torch.save(
                agent.actor_target.state_dict(),
                path + "actor_target_" + str(i) + ".pth",
            )
            torch.save(
                agent.critic_target.state_dict(),
                path + "critic_target_" + str(i) + ".pth",
            )

    def load(self, path="./saved_agents/"):
        for i, agent in enumerate(self.agents):
            actor_local_file = torch.load(
                path + "actor_local_" + str(i) + ".pth", map_location="cpu"
            )
            actor_target_file = torch.load(
                path + "actor_target_" + str(i) + ".pth", map_location="cpu"
            )
            critic_local_file = torch.load(
                path + "critic_local_" + str(i) + ".pth", map_location="cpu"
            )
            critic_target_file = torch.load(
                path + "critic_target_" + str(i) + ".pth", map_location="cpu"
            )
            agent.actor_local.load_state_dict(actor_local_file)
            agent.actor_target.load_state_dict(actor_target_file)
            agent.critic_local.load_state_dict(critic_local_file)
            agent.critic_target.load_state_dict(critic_target_file)


class DDPG_Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, agent_id, state_size, action_size, n_agents, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = n_agents
        self.seed = random.seed(seed)
        self.agent_id = agent_id
        # DDPG-Network
        self.network = DDPGModel(n_agents, state_size, action_size, seed)
        self.actor_local = self.network.actor_local
        self.actor_target = self.network.actor_target
        self.optimizer_actor = optim.Adam(
            self.actor_local.parameters(), lr=LR_ACTOR
        )

        self.critic_target = self.network.critic_target
        self.critic_local = self.network.critic_local
        self.optimizer_critic = optim.Adam(
            self.critic_local.parameters(), lr=LR_CRITIC
        )

        # set noise
        self.noise = OUNoise(action_size, seed)
        self.eps = EPS_START
        self.t_step = 0
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            noise_weights (float): how much noise to add
            add_noise (boolean): wether too add noise or not
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action_values = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            self.noise_values = self.eps * self.noise.sample()
            action_values += self.noise_values
        return np.clip(action_values, -1, 1)

    def step(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        states_others,
        actions_others,
        next_states_others,
    ):
        self.memory.add(
            state,
            action,
            reward,
            next_state,
            done,
            states_others,
            actions_others,
            next_states_others,
        )
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random 
            # subset and learn
            if len(self.memory) > BATCH_SIZE:
                # source: Sample a random minibatch of N transitions 
                # from R
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience 
           tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r,
                                                         s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, states_others, actions_others, next_states_others = (
            experiences
        )

        # ---------------------- update critic ----------------------- #
        # get predicted next actions and Q values from target models
        self.optimizer_critic.zero_grad()
        next_actions = self.actor_target(next_states)
        next_actions_others = self.actor_target(next_states_others)

        # concatenate actions next _actions, states and next states of 
        # all agents
        actions_all = torch.cat((actions, actions_others), dim=1).to(device)
        next_actions_all = torch.cat(
            (next_actions, next_actions_others), dim=1
        ).to(device)
        states_all = torch.cat((states, states_others), dim=1).to(device)
        next_states_all = torch.cat(
            (next_states, next_states_others), dim=1
        ).to(device)

        q_targets_next = self.critic_target(
            next_states_all, next_actions_all
        ).squeeze()
        # compute Q targets for current states (y_i)
        q_expected = self.critic_local(states_all, actions_all).squeeze()
        # q_targets = reward of this timestep + discount * Q(st+1,at+1) 
        # from target network
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        # compute critic loss
        critic_loss = F.mse_loss(q_expected, q_targets.detach())
        # minimize loss
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # ----------------------- update actor ----------------------- #
        # compute actor loss
        actions_pred_agent = self.actor_local(states)
        actions_pred_others = self.actor_local(states_others)
        actions_pred = torch.cat(
            (actions_pred_agent, actions_pred_others.detach()), dim=1
        ).to(device)
        actor_loss = -self.critic_local(states_all, actions_pred).mean()
        # minimize loss
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        # ------------------- update target network ------------------ #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

        # --------------------- update noise --------------------------#
        self.eps *= EPS_DECAY
        self.eps = max(self.eps, EPS_END)
        self.noise.reset()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=[
                "state",
                "action",
                "reward",
                "next_state",
                "done",
                "states_others",
                "actions_others",
                "next_states_others",
            ],
        )
        self.seed = random.seed(seed)

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        states_others,
        actions_others,
        next_states_others,
    ):
        """Add a new experience to memory."""
        e = self.experience(
            state,
            action,
            reward,
            next_state,
            done,
            states_others,
            actions_others,
            next_states_others,
        )
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = (
            torch.from_numpy(
                np.stack([e.state for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        actions = (
            torch.from_numpy(
                np.stack([e.action for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        rewards = (
            torch.from_numpy(
                np.stack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        next_states = (
            torch.from_numpy(
                np.stack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        dones = (
            torch.from_numpy(
                np.stack(
                    [e.done for e in experiences if e is not None]
                ).astype(np.uint8)
            )
            .float()
            .to(device)
        )
        states_others = (
            torch.from_numpy(
                np.stack(
                    [e.states_others for e in experiences if e is not None]
                ).astype(np.float)
            )
            .float()
            .squeeze()
            .to(device)
        )
        actions_others = (
            torch.from_numpy(
                np.stack(
                    [e.actions_others for e in experiences if e is not None]
                ).astype(np.float)
            )
            .float()
            .squeeze()
            .to(device)
        )
        next_states_others = (
            torch.from_numpy(
                np.stack(
                    [
                        e.next_states_others
                        for e in experiences
                        if e is not None
                    ]
                ).astype(np.float)
            )
            .float()
            .squeeze()
            .to(device)
        )

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            states_others,
            actions_others,
            next_states_others,
        )

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
