from torch import nn
import torch
import numpy as np

from collections import deque
import os
import itertools

import random
import cv2

from trainer import Trainer

class PolicyNetwork1(nn.Module):
    '''
    Policy network for DQN
    '''
    def __init__(self, env, device=None):
        super().__init__()

        self.need_flatten_input = True

        num_inputs = int(np.prod(env.observation_space.shape))
        num_actions = 4

        self.device = device
        self.network = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.Tanh(),
            nn.Linear(128, num_actions),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.network(x)

class PolicyNetwork2(nn.Module):
    '''
    Policy network for DQN
    '''

    def __init__(self, env, device=None):
        super().__init__()

        self.need_flatten_input = False
        self.device = device

        # Get the shape of the input
        obs_shape = env.observation_space.shape
        output_size = 4

        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        conv_out_size = 32 * obs_shape[0] * obs_shape[1]

        # Fully connected layers
        self.fc_net = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        # Add channel dimension for Conv2D input (batch, channels, height, width)
        x = x.unsqueeze(1)

        # Pass through conv layers
        conv_out = self.conv_net(x)
        # Flatten for FC layers
        conv_out = conv_out.view(conv_out.size(0), -1)
        # Pass through fully connected layers
        return self.fc_net(conv_out)



class DQNTrainer(Trainer):
    '''
    DQN trainer
    '''
    def __init__(self, env, model,
                 gamma=0.99, batch_size=32, buffer_size=50000,
                 epsilon_start=0.5, epsilon_end=0.02, epsilon_decay=100000,
                 target_update_freq=1000, min_replay_size=1000, device=None,
                 model_save_freq=10000):

        super().__init__()

        # Load hyperparameters for base DQN
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.min_replay_size = min_replay_size


        # Load model save frequency
        self.model_save_freq = model_save_freq

        # Load environment
        self.env = env

        # Load device (CPU or GPU)
        self.device = device

        # Initialize policy networks
        self.online_net = model.to(device)
        self.target_net = model.to(device)

        #Initialize optimizer
        self.optimizer = torch.optim.Adam(self.online_net.parameters())

    def load_model(self, model_path: str) -> None:
        '''
        Load existing model
        @param model_path: Path to the model
        @return: None
        '''
        if os.path.exists(model_path):
            self.online_net.load_state_dict(torch.load(model_path))
            self.target_net.load_state_dict(torch.load(model_path))
        else :
            raise ValueError("Model path does not exist")

    def train(self, num_iterations: int) -> None:
        '''
        Train the model
        '''

        # Initialize replay and reward buffers
        self.replay_buffers = [deque(maxlen=self.buffer_size) for _ in range(self.env.nb_agents)]
        self.reward_buffers = [deque([0.0], maxlen=100) for _ in range(self.env.nb_agents)]


        # Initialize episode reward
        episode_rewards = [0.0 for _ in range(self.env.nb_agents)]
        episode_steps = 0

        # Initialize observations
        observations, _ = self.env.reset()

        # Pre-populate replay buffer with random actions
        for _ in range(self.min_replay_size):
            # Sample random actions for each agent
            actions = [self.env.action_space.sample() for _ in range(self.env.nb_agents)]
            # Step the environment
            next_observations, rewards, done, truncated, infos = self.env.step(actions)
            # Store transition in replay buffers
            for agent_index in range(self.env.nb_agents):
                transition = (observations[agent_index], actions[agent_index], rewards[agent_index], next_observations[agent_index], done)
                self.replay_buffers[agent_index].append(transition)

            observations = next_observations

            if done :
                observations, _ = self.env.reset()

        # Main training loop
        observations, _ = self.env.reset()

        for step in itertools.count():

            if step >= num_iterations:
                break

            actions = [0 for _ in range(self.env.nb_agents)]

            # Gather the actions for the different agents
            for agent_index in range(self.env.nb_agents):
                # Epsilon greedy policy
                epsilon = np.interp(step, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])
                random_sample = np.random.random()

                if random_sample < epsilon:
                    actions[agent_index] = self.env.action_space.sample()
                else:
                    # Convert obs to tensor
                    obs_tensor = torch.as_tensor(observations[agent_index], dtype=torch.float32, device=self.device)
                    # Get the action from the online network (forwards pass + argmax)
                    actions[agent_index] = torch.argmax(self.online_net(obs_tensor.unsqueeze(0))).item()

            # Step the environment
            next_observations, rewards, done, truncated , info = self.env.step(actions)

            # Store transition in replay buffer
            for agent_index in range(self.env.nb_agents):
                transition = (observations[agent_index], actions[agent_index], rewards[agent_index], next_observations[agent_index], done)
                self.replay_buffers[agent_index].append(transition)
                episode_rewards[agent_index] += rewards[agent_index]

            # Update observations
            observations = next_observations
            episode_steps += 1

            if np.max(np.mean(self.reward_buffers)) > -10000:
                img = self.env.render(mode='rgb_array')
                cv2.imshow('Pacman', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if done:
                print(f"Total steps: {step} Episode max reward: {np.max(episode_rewards):.2f} EP_STEPS: {episode_steps}")
                observations, _ = self.env.reset()
                for agent_index in range(self.env.nb_agents):
                    self.reward_buffers[agent_index].append(episode_rewards[agent_index])
                episode_rewards = [0.0 for _ in range(self.env.nb_agents)]
                episode_steps = 0



            # Train the model
            self._update_model()

            # Update target network
            if step % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())

            # Save model every N steps
            if step % self.model_save_freq == 0:
                torch.save(self.online_net.state_dict(), "pacman_torch_model_{}.pth".format(step))



    def _update_model(self):
        '''
        Update the model
        '''
        # Sample a batch of transitions
        random_buffer_index = random.randint(0, self.env.nb_agents - 1)
        transitions_batch = random.sample(self.replay_buffers[random_buffer_index], self.batch_size)

        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*transitions_batch)

        # convert to array
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        # Convert to tensors
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # Compute Q-values
        q_values = self.online_net(states)
        next_q_values = self.target_net(next_states)

        # Compute target Q-values
        target_q_values = rewards + self.gamma * torch.max(next_q_values, dim=1)[0] * (1 - dones)

        # Compute loss
        action_q_values = q_values.gather(dim=1, index=actions.unsqueeze(-1))
        loss = nn.functional.smooth_l1_loss(action_q_values, target_q_values.unsqueeze(-1))

        # Zero gradients
        self.optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update the model
        self.optimizer.step()



