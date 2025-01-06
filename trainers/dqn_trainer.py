import torch
import numpy as np
from collections import deque
import os
import itertools
import random
import cv2
from trainers.trainer import Trainer
from torch import nn

class DQNTrainer(Trainer):
    def __init__(self, env, model,
                 gamma=0.99, 
                 batch_size=64,
                 buffer_size=500000,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=50000,
                 target_update_freq=1000,
                 min_replay_size=5000,
                 device=None,
                 model_save_freq=10000,
                 show_gameplay_freq=-1,
                 use_double_dqn=True, 
                 max_steps_env=1000):
        
        super().__init__()
        
        # DQN parameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.min_replay_size = min_replay_size
        self.model_save_freq = model_save_freq
        self.show_gameplay_freq = show_gameplay_freq
        
        # Environment and device setup
        self.env = env
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.online_net = model.to(self.device)
        self.target_net = model.to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        
        # Double DQN
        self.use_double_dqn = use_double_dqn
        
        # Replay buffer and priorities
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.priority_weights = deque(maxlen=self.buffer_size)

        # Add PER hyperparameters
        self.priority_alpha = 0.6  # How much prioritization to use
        self.priority_beta = 0.4   # Importance sampling weight
        self.priority_beta_increment = 0.001
        self.max_priority = 1.0

        self.n_steps = 3  # Number of steps for n-step return

        self.env.max_steps = max_steps_env       
        # Optimizer
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=0.0001)
        
    def _get_priorities(self, td_errors):
        return (np.abs(td_errors) + 1e-6) ** self.priority_alpha
    
    def calculate_n_step_return(self, rewards, next_value, dones):
        n_step_return = next_value
        for i in reversed(range(len(rewards))):
            n_step_return = rewards[i] + self.gamma * (1 - dones[i]) * n_step_return
        return n_step_return
    
    def load_model(self, model_path: str) -> None:
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.online_net.load_state_dict(state_dict)
            self.target_net.load_state_dict(state_dict)
        else:
            raise ValueError(f"Model path {model_path} does not exist")
            
    def show_gameplay(self, img):
        cv2.imshow('Game', img)
        return cv2.waitKey(1) != ord('q')
    
    def train(self, num_iterations: int) -> None:
        # Initialize buffers
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.reward_buffers = [deque([0.0], maxlen=100) for _ in range(self.env.nb_agents)]
        
        # Training metrics
        episode_rewards = [0.0 for _ in range(self.env.nb_agents)]
        episode_steps = 0
        episode_count = 0
        best_reward = float('-inf')
        
        # Initial reset
        observations, _ = self.env.reset()
        
        for step in itertools.count():
            if step >= num_iterations:
                break
                
            # Get actions for all agents
            actions = []
            epsilon = np.interp(step, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])
            
            for agent_index in range(self.env.nb_agents):
                if random.random() < epsilon:
                    actions.append(self.env.action_space.sample())
                else:
                    map_obs, info_obs = observations[agent_index]
                    map_state = torch.tensor(map_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                    info_state = torch.tensor(info_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        q_values = self.online_net(map_state, info_state)
                        action = torch.argmax(q_values).item()
                    actions.append(action)
            
            # Environment step
            next_observations, rewards, done, truncated, info = self.env.step(actions)
            
            # Store transitions
            for agent_index in range(self.env.nb_agents):
                transition = (observations[agent_index], actions[agent_index], 
                            rewards[agent_index], next_observations[agent_index], done)
                self.replay_buffer.append(transition)
                episode_rewards[agent_index] += rewards[agent_index]
            
            observations = next_observations
            episode_steps += 1
            
            # Display gameplay if needed
            if self.show_gameplay_freq != -1 and episode_count % self.show_gameplay_freq == 0:
                img = self.env.render(mode="rgb_array")
                if not self.show_gameplay(img):
                    break
            
            # Episode termination
            if done:
                max_reward = np.max(episode_rewards)
                if max_reward > best_reward:
                    best_reward = max_reward
                    torch.save(self.online_net.state_dict(), "best_model.pth")
                
                print(f"Step: {step} Episode: {episode_count} "
                      f"Reward: {max_reward:.2f} Best: {best_reward:.2f} "
                      f"Steps: {episode_steps} Epsilon: {epsilon:.3f}")
                
                observations, _ = self.env.reset()
                for agent_index in range(self.env.nb_agents):
                    self.reward_buffers[agent_index].append(episode_rewards[agent_index])
                episode_rewards = [0.0 for _ in range(self.env.nb_agents)]
                episode_steps = 0
                episode_count += 1
            
            # Model updates
            if step >= self.min_replay_size:
                self._update_model()
            
            # Target network update
            if step % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())
            
            # Regular model saving
            if step % self.model_save_freq == 0:
                torch.save(self.online_net.state_dict(), f"model_step_{step}.pth")
    
    def _update_model(self):
        # Check if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return
            
        # Ensure priority weights match buffer size
        while len(self.priority_weights) < len(self.replay_buffer):
            self.priority_weights.append(1.0)  # Default priority
            
        # PER sampling with matching sizes
        if len(self.priority_weights) > 0:
            probs = np.array(list(self.priority_weights)[:len(self.replay_buffer)])
            probs = probs / np.sum(probs)  # Normalize
            indices = np.random.choice(len(self.replay_buffer), 
                                    size=self.batch_size, 
                                    p=probs)
        else:
            indices = np.random.randint(0, len(self.replay_buffer), size=self.batch_size)

        # Get batch
        transitions = [self.replay_buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        # Process states
        info_obs = np.array([info for _, info in states])
        map_obs = np.array([map for map, _ in states])
        info_next_obs = np.array([info for _, info in next_states])
        map_next_obs = np.array([map for map, _ in next_states])
        
        # Convert to tensors
        map_obs_tensor = torch.tensor(map_obs, dtype=torch.float32).to(self.device)
        info_obs_tensor = torch.tensor(info_obs, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        map_next_obs_tensor = torch.tensor(map_next_obs, dtype=torch.float32).to(self.device)
        info_next_obs_tensor = torch.tensor(info_next_obs, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Normalize states and clip rewards
        map_obs_tensor = map_obs_tensor / 255.0
        map_next_obs_tensor = map_next_obs_tensor / 255.0
        rewards = torch.clamp(rewards, -1, 1)
        
        # Q-values computation
        current_q_values = self.online_net(map_obs_tensor, info_obs_tensor)
        
        with torch.no_grad():
            if self.use_double_dqn:
                next_actions = self.online_net(map_next_obs_tensor, 
                                            info_next_obs_tensor).max(1)[1]
                next_q_values = self.target_net(map_next_obs_tensor, 
                                              info_next_obs_tensor)
                next_q_values = next_q_values.gather(1, 
                                                   next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q_values = self.target_net(map_next_obs_tensor, 
                                              info_next_obs_tensor).max(1)[0]
        
        # Compute targets and loss
        expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        action_q_values = current_q_values.gather(1, actions.unsqueeze(1))
        
        # Update priorities
        td_errors = expected_q_values - action_q_values.squeeze()
        new_priorities = self._get_priorities(td_errors.detach().cpu().numpy())
        for idx, priority in zip(indices, new_priorities):
            if idx < len(self.priority_weights):
                self.priority_weights[idx] = priority
            else:
                self.priority_weights.append(priority)
        
        # Compute loss and update
        loss = nn.functional.huber_loss(action_q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10)
        self.optimizer.step()