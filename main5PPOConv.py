import time
from torch import nn, optim
from torch.distributions import Categorical
import torch
import numpy as np
import random
from collections import deque
import itertools
import cv2
from pacman_game import PacmanEnv

# Hyperparamètres
GAMMA = 0.99
CLIP_PARAM = 0.2
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
MAX_STEPS = 1000

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Fonction utilitaire pour normaliser les dimensions des états
def preprocess_state(state):
    state = np.array(state)
    if state.ndim == 2:  # Si c'est [height, width]
        state = np.expand_dims(state, axis=0)  # Ajoute la dimension pour les canaux
    return torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # Ajoute la dimension batch

# Modèle Actor-Critic avec CNN pour PPO
class ActorCritic(nn.Module):
    def __init__(self, obs_shape, num_actions):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Calculer dynamiquement la taille de sortie de conv_net
        conv_out_size = self.get_conv_output_size((1, obs_shape[0], obs_shape[1]))

        self.shared_net = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Tanh()
        )
        self.policy_head = nn.Linear(512, num_actions)
        self.value_head = nn.Linear(512, 1)

    def get_conv_output_size(self, shape):
        dummy_input = torch.zeros(1, *shape)
        output = self.conv_net(dummy_input)
        return int(np.prod(output.size()))

    def forward(self, x):
        assert x.ndim == 4, f"Expected input of shape [batch, channel, height, width], got {x.shape}"
        x = self.conv_net(x)
        # print("After conv_net:", x.shape)
        x = x.view(x.size(0), -1)  # Aplatir les sorties convolutionnelles
        # print("After flattening:", x.shape)
        shared_out = self.shared_net(x)
        # print("After shared_net:", shared_out.shape)
        logits = self.policy_head(shared_out)
        value = self.value_head(shared_out)
        return Categorical(logits=logits), value

# Fonction pour calculer les retours (returns)
def compute_returns(rewards, masks, next_value, gamma=GAMMA):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * masks[step] * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32, device=device).unsqueeze(1)

# Fonction de mise à jour PPO
def ppo_update(policy, optimizer, trajectories):
    states = torch.cat([t[0] for t in trajectories]).to(device)
    actions = torch.cat([t[1] for t in trajectories]).to(device)
    old_log_probs = torch.cat([t[2] for t in trajectories]).to(device)
    returns = torch.cat([t[3] for t in trajectories]).to(device)
    advantages = returns - torch.cat([t[4] for t in trajectories]).squeeze().to(device)

    dist, values = policy(states)
    new_log_probs = dist.log_prob(actions)
    ratio = (new_log_probs - old_log_probs.detach()).exp()

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - CLIP_PARAM, 1.0 + CLIP_PARAM) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = nn.functional.mse_loss(values.squeeze(-1), returns.squeeze(-1))

    optimizer.zero_grad()
    (policy_loss + value_loss).backward()
    optimizer.step()

# Environnement
env = PacmanEnv("pacman_game/res/level11/", flatten_observation=False)
obs_shape = env.observation_space.shape
num_actions = np.prod(env.action_space.nvec)

# Modèle et optimiseur
policy = ActorCritic(obs_shape, num_actions).to(device)
model_path ="ppo_model_episode_3000.pth"
policy.load_state_dict(torch.load(model_path))

optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

# Boucle principale d'entraînement
trajectories = []
reward_buffer = deque([0.0], maxlen=100)

for i_episode in itertools.count(1):
    episode_reward = 0.0
    observations, _ = env.reset()
    # print("Raw observations after reset:", np.array(observations).shape)

    # Préparation de l'état
    states = preprocess_state(observations)
    # print("Final input shape to conv_net:", states.shape)

    for t in range(MAX_STEPS):
        dist, value = policy(states)
        action = dist.sample()

        next_states, rewards, done, truncated, info = env.step([action.item()])
        img = env.render(mode="rgb_array")
        # cv2.imshow("Pacman", img)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        if i_episode % 50 == 0:
            
            img = env.render(mode="rgb_array")
            cv2.imshow("Pacman", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        reward = rewards[0]
        next_states = preprocess_state(next_states)

        trajectories.append((
            states,
            action,
            dist.log_prob(action),
            torch.tensor([reward], dtype=torch.float32),
            value
        ))

        states = next_states
        episode_reward += reward

        if done or truncated:
            # print(f"Game over! Done: {done}, Truncated: {truncated}, Reward: {episode_reward}, Moyenne = {np.mean(reward_buffer)}")

            # Réinitialiser l'environnement
            observations, _ = env.reset()
            states = preprocess_state(observations)

            # Calcul des retours
            _, next_value = policy(states)
            returns = compute_returns(
                [t[3].item() for t in trajectories],
                [1 - (done or truncated) for _ in trajectories],
                next_value.detach().item()
            )

            for idx, t in enumerate(trajectories):
                trajectories[idx] = (*t[:3], returns[idx], t[4])

            ppo_update(policy, optimizer, trajectories)
            trajectories.clear()

            print(f"Episode {i_episode}\tReward: {episode_reward}, Truncated: {truncated}\nMoyenne = {np.mean(reward_buffer)}")
            reward_buffer.append(episode_reward)
            break

    if i_episode % 250 == 0:
        torch.save(policy.state_dict(), f"ppo_model_episode_{i_episode}.pth")