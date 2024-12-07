from torch import nn
import torch
import numpy as np
import torch.nn.init as init
from torch.distributions import Categorical
'''
Collection of neural networks used in our PacmanAI50 project
'''

class ConvDQN(nn.Module):
    '''
    Convolutional DQN
    '''
    def __init__(self, environment):
        super(ConvDQN, self).__init__()

        # Get information about the environment
        self.input_shape = environment.observation_space.shape
        self.output_shape = environment.possible_actions

        # Convolutional layers
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Fully connected layers
        self.fc_net = nn.Sequential(
            nn.Linear(32 * self.input_shape[0] * self.input_shape[1], 256),
            nn.ReLU(),
            nn.Linear(256, self.output_shape)
        )

    def forward(self, x):
        # x should be reshaped to (batch, channels, height, width)
        # in our case (batch,31,28) -> (1, batch, 31, 28)
        x = x.unsqueeze(1)

        # Pass through conv layers
        conv_out = self.conv_net(x)
        # Flatten for FC layers
        conv_out = conv_out.view(conv_out.size(0), -1)
        # Pass through fully connected layers
        return self.fc_net(conv_out)


class NNDQN(nn.Module):
    def __init__(self, environment):
        super(NNDQN, self).__init__()

        # Get information about the environment
        self.input_shape = environment.observation_space.shape
        self.output_shape = environment.possible_actions

        # Fully connected layers
        self.fc_net = nn.Sequential(
            nn.Linear(int(np.prod(self.input_shape)), 256),
            nn.ReLU(),
            nn.Linear(256, self.output_shape),
            nn.ReLU(),
        )

    def forward(self, x):
        # x is of shape (1,31,28)
        x = x.unsqueeze(1)
        # Flatten x to (1, 31*28)
        x = x.view(x.size(0), -1)
        return self.fc_net(x)



class NNPPOActor(nn.Module):
    def __init__(self, environment):
        super(NNPPOActor, self).__init__()

        # Get information about the environment
        self.input_shape = environment.observation_space.shape
        self.output_shape = environment.possible_actions

        # Fully connected layers
        self.fc_net = nn.Sequential(
            nn.Linear(int(np.prod(self.input_shape)), 256),
            nn.ReLU(),
            nn.Linear(256, self.output_shape),
            nn.ReLU(),
        )

    def forward(self, x):
        # x is of shape (1,31,28)
        x = x.unsqueeze(1)
        # Flatten x to (1, 31*28)
        x = x.view(x.size(0), -1)

        # Pass through fully connected layers
        return self.fc_net(x)

class NNPPOCritic(nn.Module):
    def __init__(self, environment):
        super(NNPPOCritic, self).__init__()

        # Get information about the environment
        self.input_shape = environment.observation_space.shape
        self.output_shape = environment.possible_actions

        # Fully connected layers
        self.fc_net = nn.Sequential(
            nn.Linear(int(np.prod(self.input_shape)), 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        # x is of shape (1,31,28)
        x = x.unsqueeze(1)
        # Flatten x to (1, 31*28)
        x = x.view(x.size(0), -1)

        # Pass through fully connected layers
        return self.fc_net(x)


class ConvPPOActor(nn.Module):
    '''
    Convolutional DQN
    '''
    def __init__(self, environment):
        super(ConvPPOActor, self).__init__()

        # Get information about the environment
        self.input_shape = environment.observation_space.shape
        self.output_shape = environment.possible_actions

        # Convolutional layers
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Fully connected layers
        self.fc_net = nn.Sequential(
            nn.Linear(32 * self.input_shape[0] * self.input_shape[1], 256),
            nn.ReLU(),
            nn.Linear(256, self.output_shape)
        )

    def forward(self, x):
        # x should be reshaped to (batch, channels, height, width)
        # in our case (batch,31,28) -> (1, batch, 31, 28)
        x = x.unsqueeze(1)

        # Pass through conv layers
        conv_out = self.conv_net(x)
        # Flatten for FC layers
        conv_out = conv_out.view(conv_out.size(0), -1)
        # Pass through fully connected layers
        return self.fc_net(conv_out)


class ConvPPOCritic(nn.Module):
    '''
    Convolutional DQN
    '''
    def __init__(self, environment):
        super(ConvPPOCritic, self).__init__()

        # Get information about the environment
        self.input_shape = environment.observation_space.shape
        self.output_shape = environment.possible_actions

        # Convolutional layers
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Fully connected layers
        self.fc_net = nn.Sequential(
            nn.Linear(32 * self.input_shape[0] * self.input_shape[1], 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # x should be reshaped to (batch, channels, height, width)
        # in our case (batch,31,28) -> (1, batch, 31, 28)
        x = x.unsqueeze(1)

        # Pass through conv layers
        conv_out = self.conv_net(x)
        # Flatten for FC layers
        conv_out = conv_out.view(conv_out.size(0), -1)
        # Pass through fully connected layers
        return self.fc_net(conv_out)


# Neural Network for the policy and value functions
class NNActorCritic(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(NNActorCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.policy = nn.Linear(128, n_actions)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        # x is of shape (1,31,28)
        x = x.unsqueeze(1)
        # Flatten the input
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.policy(x), self.value(x)

class ConvActorCritic(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super(ConvActorCritic, self).__init__()

        # Convolutional layers
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        conv_out_size = self.get_conv_output_size((1, obs_shape[0], obs_shape[1]))

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.policy = nn.Linear(256, n_actions)
        self.value = nn.Linear(256, 1)

    def get_conv_output_size(self, shape):
        '''
        Get the output size of the convolutional layers by passing a dummy input and checking the output size
        @param shape: tuple, shape of the input
        '''
        dummy_input = torch.zeros(1, *shape)
        output = self.conv_net(dummy_input)
        return int(torch.prod(torch.tensor(output.size())))


    def forward(self, x):
        # x is of shape (1,31,28)
        x = x.unsqueeze(1)

        # Pass through conv layers
        x = self.conv_net(x)

        # Flatten the input
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return self.policy(x), self.value(x)

class ActorCritic(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super(ActorCritic, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(obs_shape[0]*obs_shape[1], 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
        )

        self.policy = nn.Linear(256, n_actions)
        self.value = nn.Linear(256, 1)

    def forward(self, x):
        # x is of shape (1,31,28)
        x = x.unsqueeze(1)

        # Flatten the input
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return self.policy(x), self.value(x)


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        init.kaiming_normal_(module.weight, nonlinearity='relu')  # For ReLU activation
        if module.bias is not None:
            init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        init.xavier_uniform_(module.weight)  # Xavier is fine for fully connected layers
        if module.bias is not None:
            init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)



# Modèle Actor-Critic avec CNN pour PPO
class ActorCriticED(nn.Module):
    def __init__(self, obs_shape, num_actions):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Calculer dynamiquement la taille de sortie de conv_net
        conv_out_size = self.get_conv_output_size((1, obs_shape[0], obs_shape[1]))
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Tanh()
        )
        self.policy = nn.Linear(512, num_actions)
        self.value = nn.Linear(512, 1)

    def get_conv_output_size(self, shape):
        dummy_input = torch.zeros(1, *shape)
        output = self.conv_net(dummy_input)
        return int(np.prod(output.size()))

    def forward(self, x):
        x = self.conv_net(x)
        # print("After conv_net:", x.shape)
        x = x.view(x.size(0), -1)  # Aplatir les sorties convolutionnelles
        # print("After flattening:", x.shape)
        shared_out = self.fc(x)
        # print("After shared_net:", shared_out.shape)
        logits = self.policy(shared_out)
        value = self.value(shared_out)
        return Categorical(logits=logits), value