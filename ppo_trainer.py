from torch import nn
from torch.distributions import MultivariateNormal
import torch
from torch.optim import Adam
import numpy as np

from collections import deque
import os
import itertools

import random
import cv2

from trainer import Trainer


class PPOTrainer(Trainer):
    def __init__(self, env, actor_model, critic_model, device=None):
        self.env = env

        dim_actions = 4

        self.actor = actor_model
        self.critic = critic_model

        self.cov_var = torch.full(size=(dim_actions,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        self.init_hyperparameters()


        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)


    def init_hyperparameters(self):
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 2000
        self.n_updates_per_iteration = 5
        self.clip = 0.2
        self.lr = 0.005
        self.gamma = 0.99

    def rollout(self):
        # Batch data
        batch_obs = [[] for _ in range(self.env.nb_agents)]     # batch observations
        batch_acts = [[] for _ in range(self.env.nb_agents)]            # batch actions
        batch_log_probs = [[] for _ in range(self.env.nb_agents)]       # log probs of each action
        batch_rews = [[] for _ in range(self.env.nb_agents)]            # batch rewards
        batch_rtgs = [[] for _ in range(self.env.nb_agents)]            # batch rewards-to-go
        batch_lens = [[] for _ in range(self.env.nb_agents)]            # episodic lengths in batch

        # Number of timesteps ran in this batch
        t = 0
        while t < self.timesteps_per_batch:
            # Rewards this episode
            ep_rews = [[] for _ in range(self.env.nb_agents)]

            observations, infos = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                # render the environment
                if True:
                    img = self.env.render(mode='rgb_array')
                    cv2.imshow('Pacman', img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Increment timesteps ran this batch
                t += 1

                # Calculate action and log prob for each agent
                actions = [[] for _ in range(self.env.nb_agents)]
                log_probs = [[] for _ in range(self.env.nb_agents)]
                actions_to_play = []

                for i, obs in enumerate(observations):
                    batch_obs[i].append(obs)
                    action, log_prob = self.get_action(obs)
                    actions[i].append(action)
                    log_probs[i].append(log_prob)

                    # Add the max action index to the list of actions to play
                    actions_to_play.append(np.argmax(action))

                # Step environment
                next_observations, rewards, done, truncated, infos = self.env.step(actions_to_play)

                # Collect rewards
                for i in range(self.env.nb_agents):
                    ep_rews[i].append(rewards)
                    batch_acts[i].append(actions)
                    batch_log_probs[i].append(log_probs)

                if done:
                    break

                # Update observations for next timestep
                observations = next_observations

            print(f"Episode done. Timesteps run: {ep_t + 1}, Episode reward: {np.sum(ep_rews[0])}")

            for i in range(self.env.nb_agents):
                # Collect episodic length and rewards
                batch_lens[i].append(ep_t + 1)
                batch_rews[i].append(ep_rews[i])

        batch_rtgs = [[] for _ in range(self.env.nb_agents)]
        for i in range(self.env.nb_agents):
            # Calculate reward-to-go
            batch_rtgs[i] = self.compute_rtgs(batch_rews[i])

        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []
        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        # Convert the rewards-to-go into a tensor
        batch_rtgs = np.array(batch_rtgs)
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float).squeeze()
        return batch_rtgs

    def get_action(self, obs):
        # Query the actor network for a mean action.
        # Same thing as calling self.actor.forward(obs)

        # Convert the observation to a tensor
        obs = torch.tensor(obs, dtype=torch.float)
        mean = self.actor(obs.unsqueeze(0))

        # Create our Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log prob of that action
        # Note that I'm calling detach() since the action and log_prob
        # are tensors with computation graphs, so I want to get rid
        # of the graph and just convert the action to numpy array.
        # log prob as tensor is fine. Our computation graph will
        # start later down the line.
        return action.detach().numpy(), log_prob.detach()


    def evaluate(self, batch_obs, batch_acts):
        # Query critic network for a value V for each obs in batch_obs.
        # convert the observations to tensors
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most
        # recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        # Return predicted values V and log probs log_probs
        return V, log_probs

    def train(self, total_timesteps=1000000):
        timesteps_done = 0

        while timesteps_done < total_timesteps:
            print(f"Total timesteps done: {timesteps_done}")
            print("Collecting rollouts...")
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            print("Rollouts collected. Updating parameters...")
            # Here we have the observations, actions for ALL agents. We will only use the actions of the first agent to train the actor and critic (for now)
            # That is not optimal (e.g. we lose half of the training when n_agents = 2) and should be changed later on
            batch_obs = np.array(batch_obs[0])
            batch_acts = np.array(batch_acts[0])
            batch_log_probs = np.array(batch_log_probs[0])
            batch_lens = np.array(batch_lens[0])
            batch_rtgs = batch_rtgs[0]

            # Reshape data as tensors in the shape specified before returning
            batch_obs = torch.tensor(batch_obs, dtype=torch.float)
            batch_acts = torch.tensor(batch_acts, dtype=torch.float)
            batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

            # Calculate V_{phi, k}
            V, _ = self.evaluate(batch_obs, batch_acts)

            # Calculate advantage
            A_k = batch_rtgs - V.detach()

            # Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for i in range(self.n_updates_per_iteration):
                print(f"Updating actor and critic networks...{i}/{self.n_updates_per_iteration}")
                # Calculate pi_theta(a_t | s_t)
                _, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # Calculate ratios
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # Calculate actor and critic losses
                actor_loss = (-torch.min(surr1, surr2)).mean()

                # Calculate gradients and perform backward propagation for actor
                # network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                critic_loss = nn.MSELoss()(V, batch_rtgs)
                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

            timesteps_done += np.sum(batch_lens)
