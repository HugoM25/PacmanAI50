import torch
from torch import nn, optim
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np
import itertools
import cv2
from pacman_game import PacEnv

# Définition de la politique PPO avec un réseau de neurones simple
class PPOPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_inputs, 128),  # Première couche linéaire
            nn.Tanh(),                 # Fonction d'activation Tanh
            nn.Linear(128, 128),         # Deuxième couche linéaire
            nn.Tanh(),                 # Fonction d'activation Tanh
        )
        self.policy_head = nn.Linear(128, num_actions)  # Tête de la politique pour les actions
        self.value_head = nn.Linear(128, 1)             # Tête de la valeur pour l'évaluation des états

    def forward(self, x):
        x = self.network(x)
        return Categorical(logits=self.policy_head(x)), self.value_head(x)

# Fonction pour calculer les retours (returns)
def compute_returns(next_value, rewards, masks, gamma=0.4):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return torch.cat(returns)

# Fonction pour mettre à jour la politique PPO
def ppo_update(policy, optimizer, trajectories, clip_param=0.2):
    states = torch.cat([t[0] for t in trajectories])
    actions = torch.cat([t[1] for t in trajectories])
    old_probs = torch.cat([t[2] for t in trajectories])
    returns = torch.cat([t[3] for t in trajectories])
    masks = torch.cat([t[4] for t in trajectories])

    dist, values = policy(states)
    new_probs = dist.log_prob(actions)
    ratio = torch.exp(new_probs - old_probs.detach())
    advantages = returns - values.detach()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages

    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = (returns - values).pow(2).mean()

    optimizer.zero_grad()
    (policy_loss + value_loss).backward()
    optimizer.step()

# Fonction principale pour exécuter l'entraînement PPO
def main():
    env = PacEnv("pacman_game/res/level0/", flatten_observation=True)
    policy = PPOPolicy(np.prod(env.observation_space.shape), np.prod(env.action_space.nvec))
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    trajectories = []  # Initialisation de la liste des trajectoires
    steps_since_reset = 0  # Compteur de coups depuis la dernière réinitialisation

    for i_episode in itertools.count(1):
        states, _ = env.reset()
        states = np.array(states).flatten()
        state = torch.from_numpy(states).float().unsqueeze(0)
        episode_reward = 0

        while True:
            dist, value = policy(state)
            action = dist.sample()

            next_states, rewards, done, truncated, info = env.step([action.item()])
            img = env.render(mode='rgb_array')
            cv2.imshow('Pacman', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            reward = rewards[0] if isinstance(rewards, list) else rewards
            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            mask_tensor = torch.tensor([1.0 - float(done)], dtype=torch.float32)

            next_states = np.array(next_states).flatten()
            next_state = torch.from_numpy(next_states).float().unsqueeze(0)
            episode_reward += reward

            trajectories.append((state, action, dist.log_prob(action), reward_tensor, mask_tensor, value))
            state = next_state
            steps_since_reset += 1  # Incrémenter le compteur de coups

            # Vérifier si le nombre de coups atteint 1000
            if steps_since_reset >= 1000 or done:
                print(f'Episode {i_episode}\tLast reward: {episode_reward:.2f}')
                break

        _, next_value = policy(state)
        returns = compute_returns(next_value, [t[3] for t in trajectories], [t[4] for t in trajectories])

        ppo_update(policy, optimizer, trajectories, clip_param=0.2)
        trajectories.clear()  # Nettoyage des trajectoires après chaque épisode
        steps_since_reset = 0  # Réinitialisation du compteur de coups

        if i_episode % 10 == 0:
            torch.save(policy.state_dict(), f"pacman_ppo_model_episode_{i_episode}.pth")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
