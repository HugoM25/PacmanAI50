from torch import nn
import torch
import numpy as np

from collections import deque
import os
import itertools

import random
import cv2

from trainers.trainer import Trainer

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

        # Main training loop
        observations, _ = self.env.reset()

        for step in itertools.count():

            # Train until we reach the number of iterations
            if step >= num_iterations:
                break

           
            # Gather the actions for the different agents
            actions = [0 for _ in range(self.env.nb_agents)]

            for agent_index in range(self.env.nb_agents):
                
                # Epsilon greedy policy
                epsilon = np.interp(step, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])
                random_sample = np.random.random()

                if random_sample < epsilon:
                    # Random action
                    actions[agent_index] = self.env.action_space.sample()
                else:
                    # Convert obs to tensor
                    map_obs, info_obs = observations[agent_index]
                    map_state_tensor = torch.tensor(map_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                    info_state_tensor = torch.tensor(info_obs, dtype=torch.float32).unsqueeze(0).to(self.device)

                    actions[agent_index] = torch.argmax(self.online_net(map_state_tensor, info_state_tensor)).item()

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

            if np.max(np.mean(self.reward_buffers)) > -1000000:
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


            # Train the model if we have enough transitions
            if step >= self.min_replay_size:
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
        info_obs = np.array([info for _, info in states])
        map_obs = np.array([map for map, _ in states])

        info_next_obs = np.array([info for _, info in next_states])
        map_next_obs = np.array([map for map, _ in next_states])

        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        # Convert to tensors
        info_obs_tensor = torch.tensor(info_obs, dtype=torch.float32).to(self.device)
        map_obs_tensor = torch.tensor(map_obs, dtype=torch.float32).to(self.device).squeeze(0)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        info_obs_next_tensor = torch.tensor(info_next_obs, dtype=torch.float32).to(self.device)
        map_obs_next_tensor = torch.tensor(map_next_obs, dtype=torch.float32).to(self.device).squeeze(0)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # Compute Q-values
        q_values = self.online_net(map_obs_tensor, info_obs_tensor)
        next_q_values = self.target_net(map_obs_next_tensor, info_obs_next_tensor)

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



