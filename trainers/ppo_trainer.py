import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import cv2

from collections import deque

from trainers.trainer import Trainer


class PPOTrainer(Trainer):
    def __init__(self, env, model, device, optimizer, clip_epsilon, gamma, entropy_coef, value_coef, epochs=4, batch_size=32, n_steps=128):
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

    def compute_advantages(self, rewards, dones, values, next_value):
        advantages = []
        advantage = 0
        for i in reversed(range(len(rewards))):
            td_error = rewards[i] + self.gamma * (1 - dones[i]) * next_value - values[i]
            advantage = td_error + self.gamma * (1 - dones[i]) * advantage
            advantages.insert(0, advantage)
            next_value = values[i]
        return advantages
    
    # Fonction pour calculer les retours (returns)
    # def compute_returns(rewards, masks, next_value, gamma=GAMMA):
    #     R = next_value
    #     returns = []
    #     for step in reversed(range(len(rewards))):
    #         R = rewards[step] + gamma * masks[step] * R
    #         returns.insert(0, R)
    #     return torch.tensor(returns, dtype=torch.float32, device=device).unsqueeze(1)

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
            returns = advantages + values

        for _ in range(self.epochs):
            for idx in range(0, len(states), self.batch_size):
                b_states = states[idx:idx + self.batch_size]
                b_actions = actions[idx:idx + self.batch_size]
                b_log_probs = log_probs[idx:idx + self.batch_size]
                b_advantages = advantages[idx:idx + self.batch_size]
                b_returns = returns[idx:idx + self.batch_size]

                logits, value = self.model(b_states)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratio = (new_log_probs - b_log_probs).exp()
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (b_returns - value).pow(2).mean()
                loss = policy_loss - self.value_coef * value_loss - self.entropy_coef * entropy
                #loss = policy_loss - value_loss 

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # logits, values = self.model(b_states)
                # dist = torch.distributions.Categorical(logits=logits)
                # new_log_probs = dist.log_prob(b_actions)
                # ratio = (new_log_probs - b_log_probs).exp()

                # surr1 = ratio * b_advantages
                # surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * b_advantages
                # policy_loss = -torch.min(surr1, surr2).mean()
                # value_loss = nn.functional.mse_loss(values, b_returns)

                # self.optimizer.zero_grad()
                # (policy_loss + value_loss).backward()
                # self.optimizer.step()

    def collect_trajectories(self, render=False):
        '''
        Collect trajectories for each agent
        '''
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
        for _ in range(self.env.nb_agents):
            states.append([])
            actions.append([])
            rewards.append([])
            dones.append([])
            log_probs.append([])
            values.append([])

        # Reset the environment 
        observations, _ = self.env.reset()
        done = False

        while len(rewards[0]) < self.n_steps:

            if render :
                img = self.env.render(mode='rgb_array')
                cv2.imshow('Pacman', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

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
        

            if done or truncated:
                observations, _ = self.env.reset()
        
        rewards_n_steps = [np.sum(rewards[agent_index]) for agent_index in range(self.env.nb_agents)]

        # Zip the trajectories and return them
        trajectories = []
        for agent_index in range(self.env.nb_agents):
            trajectories.append(
                zip(states[agent_index], actions[agent_index], rewards[agent_index], dones[agent_index], log_probs[agent_index], values[agent_index]))

        return trajectories, rewards_n_steps
    
    def train(self, num_iterations):

        mean_rewards_buffer = deque(maxlen=100)
        for episode in range(num_iterations):
            render = False
            if episode % 100 == 0:
                render = True

            # Collect trajectories
            trajectories, rewards = self.collect_trajectories(render=render)

            # Update the model
            for trajectory in trajectories:
                self.update_model(trajectory)

            mean_rewards_buffer.append(np.mean(rewards))
                
            print(f"Episode {episode} done with rewards: {np.mean(rewards)} mean rewards: {np.mean(mean_rewards_buffer)}")

        self.save_model()

    def save_model(self):
        torch.save(self.model.state_dict(), "model.pth")
