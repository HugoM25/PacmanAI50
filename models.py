from torch import nn
import torch
import numpy as np

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
