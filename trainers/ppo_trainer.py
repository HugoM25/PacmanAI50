import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import cv2

from collections import deque

from trainers.trainer import Trainer


class PPOTrainer(Trainer):
    def __init__(self, env, model, device, optimizer, clip_epsilon, gamma, entropy_coef, value_coef, epochs=4, batch_size=32, n_steps=2048, gae_lambda=0.95):
        self.env = env
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.gae_lambda = gae_lambda

    def compute_advantages(self, rewards, dones, values, next_value):
        returns = []
        gae = 0
        next_value = next_value
        for r, d, v in zip(reversed(rewards), reversed(dones), reversed(values)):
            delta = r + self.gamma * next_value * (1 - d) - v
            gae = delta + self.gamma * self.gae_lambda * (1 - d) * gae
            next_value = v
            returns.insert(0, gae + v)
        return returns

    def update_model(self, trajectory):
        states, actions, rewards, dones, log_probs, values = zip(*trajectory)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        log_probs = torch.tensor(log_probs, dtype=torch.float32).to(self.device)
        values = torch.tensor(values, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            next_value = self.model(states[-1].unsqueeze(0))[1].item()
            advantages = self.compute_advantages(rewards, dones, values, next_value)
            advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
            # Normalize the advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            returns = advantages + values

        # Batch update with random sampling
        indices = np.random.permutation(len(states))

        for _ in range(self.epochs):
            for idx in range(0, len(states), self.batch_size):
                end = idx + self.batch_size
                batch_indices = indices[idx:end]

                # Get the batch
                b_states = states[batch_indices]
                b_actions = actions[batch_indices]
                b_log_probs = log_probs[batch_indices]
                b_advantages = advantages[batch_indices]
                b_returns = returns[batch_indices]

                # Current policy outputs
                logits, value = self.model(b_states)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(b_actions)

                # Compute PPO losses
                ratio = (new_log_probs - b_log_probs).exp()
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * b_advantages

                # Combine loss functions
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (b_returns - value).pow(2).mean()
                entropy = dist.entropy().mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

    def render(self):
        img = self.env.render(mode='rgb_array')
        cv2.imshow('Pacman', img)
        cv2.waitKey(1) & 0xFF == ord('q')
        return              
        
    
    def train(self, max_steps):
        mean_rewards_buffer = deque(maxlen=100)
        current_step = 0
        episode_count = 0

        # Train for num_steps
        while current_step < max_steps:

            states = [ [] for _ in range(self.env.nb_agents)]
            actions = [ [] for _ in range(self.env.nb_agents)]
            rewards = [ [] for _ in range(self.env.nb_agents)]
            dones = [ [] for _ in range(self.env.nb_agents)]
            log_probs = [ [] for _ in range(self.env.nb_agents)]
            values = [ [] for _ in range(self.env.nb_agents)]

            done = False
            episode_rewards = [0 for _ in range(self.env.nb_agents)]
            observations, _ = self.env.reset()

            # Collect trajectories by playing the game 
            while len(rewards[0]) < self.n_steps and not done:
                
                current_step += 1

                # Render the environment every X episodes
                if episode_count % 50 == 0:
                    self.render()

                # self.render()
                
                actions_to_play = []

                # For each agent, get the action to play
                for agent_index, observation in enumerate(observations):
                        state_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
                        logits, value = self.model(state_tensor)
                        dist = torch.distributions.Categorical(logits=logits)
                        action = dist.sample()
                        log_prob = dist.log_prob(action)
                        actions_to_play.append(action.item())

                        states[agent_index].append(observation)
                        actions[agent_index].append(action.item())
                        log_probs[agent_index].append(log_prob.item())
                        values[agent_index].append(value.item())

                # Step the environment
                observations, rewards_earned, done, truncated, infos = self.env.step(actions_to_play)

                for agent_index, reward in enumerate(rewards_earned):
                    rewards[agent_index].append(reward)
                    dones[agent_index].append(done)

                    episode_rewards[agent_index] += reward
            
                if done or truncated:
                    observations, _ = self.env.reset()
                    episode_count += 1


                # Zip the trajectories and return them
                trajectories = []
                for agent_index in range(self.env.nb_agents):
                    trajectories.append(
                        zip(states[agent_index], actions[agent_index], rewards[agent_index], dones[agent_index], log_probs[agent_index], values[agent_index]))
            
            mean_rewards_buffer.append(np.mean(episode_rewards))    

                    
            if episode_count % 1 == 0:
                print(f"Episode: {episode_count}, Mean rewards: {np.mean(mean_rewards_buffer)}, episode rewards: {episode_rewards[0]}")
                    
            # Update the model
            for trajectory in trajectories:
                self.update_model(trajectory)

            if current_step % 50000 == 0:
                self.save_model()

        self.save_model()

    def save_model(self):
        torch.save(self.model.state_dict(), "model.pth")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
