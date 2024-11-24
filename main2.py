from torch import nn
import torch
import gymnasium as gym
import numpy as np
import random
from collections import deque
import itertools

import cv2
import os
import time

from pacman_game import PacEnv

np.bool = np.bool_  # Fix for deprecated usage of np.bool

GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 50000
EPSILON_START = 0.5
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 1000
MIN_REPLAY_SIZE = 1000

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

class PolicyNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()

        # Get the shape of the input
        obs_shape = env.observation_space.shape

        # Convolutional layers
        # self.conv_net = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU()
        # )

        # # Calculate the size of the feature map after conv layers
        # conv_out_size = 64 * obs_shape[0] * obs_shape[1]

        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # Conv1
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Conv2
            nn.ReLU()
        )
        conv_out_size = 32 * obs_shape[0] * obs_shape[1]  # Assuming no downsampling in convolutions


        # Fully connected layers
        self.fc_net = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # Output layer for Q-values
        )

    def forward(self, x):
        # Add channel dimension for Conv2D input (batch, channels, height, width)
        x = x.unsqueeze(1)
        # Pass through conv layers
        conv_out = self.conv_net(x)
        conv_out = conv_out.view(conv_out.size(0), -1)  # Flatten for FC layers
        # Pass through fully connected layers
        return self.fc_net(conv_out)

    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        q_values = self(obs_t.unsqueeze(0))

        action = torch.argmax(q_values, dim=1)[0].item()
        return action


env = PacEnv("pacman_game/res/level0/", flatten_observation=False)

info = {}

replay_buffer = deque(maxlen=BUFFER_SIZE)
reward_buffer = deque([0.0], maxlen=100)

episode_reward = 0.0

online_net = PolicyNetwork(env).to(device)
target_net = PolicyNetwork(env).to(device)

# Load the model if a pre-trained model exists
path_to_model = "pacman_torch_model_1080000.pth"
if os.path.exists(path_to_model):
    online_net.load_state_dict(torch.load(path_to_model))
    print("Model loaded successfully.")
else:
    print("No model found, starting from scratch.")

target_net.load_state_dict(online_net.state_dict())
optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4)

observations, _ = env.reset()


# Pre-populate replay buffer with random actions
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()
    next_observations, rewards, done, _, info = env.step([action])
    transition = (observations[0], action, rewards[0], next_observations[0], done)
    replay_buffer.append(transition)

    observations = next_observations

    if done:
        observations, _ = env.reset()

# Main training loop
observations, _ = env.reset()

for step in itertools.count():
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
    random_sample = random.random()

    if random_sample < epsilon:
        action = env.action_space.sample()
    else:
        action = online_net.act(observations[0])

    next_observations, rewards, done, _ , info = env.step([action])
    transition = (observations[0], action, rewards[0], next_observations[0], done)
    replay_buffer.append(transition)

    observations = next_observations
    episode_reward += rewards[0]

    # time.sleep(0.1)

    #If the reward gets over X, render the environment
    if np.mean(reward_buffer) > -10000:
        img = env.render(mode="rgb_array")
        cv2.imshow("Pacman", img)
        cv2.waitKey(1)

    if done:
        print(f"Step: {step}, Reward: {np.mean(reward_buffer)}, Pacgum eaten: {env.agents[0].pacgum_eaten}/{np.sum(env.map.type_map == 1)}, Episode reward: {episode_reward}")
        observations, info = env.reset()
        reward_buffer.append(episode_reward)
        episode_reward = 0.0



    # Only start training when there's enough data in the buffer
    if len(replay_buffer) >= BATCH_SIZE:
        # Sample a batch of transitions
        transitions = random.sample(replay_buffer, BATCH_SIZE)

        obses = np.asarray([t[0] for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rewards = np.asarray([t[2] for t in transitions])
        next_obses = np.asarray([t[3] for t in transitions])
        dones = np.asarray([t[4] for t in transitions])

        # Convert to tensors
        obses_t = torch.as_tensor(obses, dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=device).unsqueeze(-1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(-1)
        next_obses_t = torch.as_tensor(next_obses, dtype=torch.float32, device=device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(-1)

        # Compute targets
        target_q_values = target_net(next_obses_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
        targets = rewards_t + GAMMA * (1 - dones_t) * max_target_q_values

        # Compute loss
        q_values = online_net(obses_t)
        action_q_values = q_values.gather(dim=1, index=actions_t)
        loss = nn.functional.smooth_l1_loss(action_q_values, targets)

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Update target network
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    # Save the model every 20000 steps
    if step % 40000 == 0:
        torch.save(online_net.state_dict(), f"pacman_torch_model_{step}.pth")
