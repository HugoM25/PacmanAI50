from torch import nn
import torch
import numpy as np
import torch.nn.init as init
from torch.distributions import Categorical
'''
Collection of neural networks used in our PacmanAI50 project
'''

class PacmanModelDQN(nn.Module):
    '''
    Convolutional DQN
    '''
    def __init__(self, environment, n_actions):
        super(PacmanModelDQN, self).__init__()

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

    def forward(self, x, add_info_t):
        # x should be reshaped to (batch, channels, height, width)
        # in our case (batch,31,28) -> (1, batch, 31, 28)
        x = x.unsqueeze(1)

        # Pass through conv layers
        conv_out = self.conv_net(x)
        # Flatten for FC layers
        conv_out = conv_out.view(conv_out.size(0), -1)
        # Pass through fully connected layers
        return self.fc_net(conv_out)

class PacmanModelPPO(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super(PacmanModelPPO, self).__init__()

        # Convolutional layers
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

        # Calculate the output size of the conv layers
        conv_out_size = self.get_conv_output_size((1, obs_shape[0], obs_shape[1]))
        
        # Additional information size ( ex: score, position)
        additional_info_size = 7
        self.additional_info_net = nn.Sequential(
            nn.Linear(additional_info_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.policy = nn.Linear(128, n_actions)
        self.value = nn.Linear(128, 1)

    def get_conv_output_size(self, shape):
        '''
        Get the output size of the convolutional layers by passing a dummy input and checking the output size
        @param shape: tuple, shape of the input
        '''
        dummy_input = torch.zeros(1, *shape)
        output = self.conv_net(dummy_input)
        return int(torch.prod(torch.tensor(output.size())))


    def forward(self, map_obs_t, add_info_t):

        # x is of shape (1,31,28)
        map_obs_t = map_obs_t.unsqueeze(1)
        # Pass map through conv layers
        map_obs_t = self.conv_net(map_obs_t)

        # Flatten the input
        map_obs_t = map_obs_t.view(map_obs_t.size(0), -1)

        # Process additional information
        add_info_t = self.additional_info_net(add_info_t)

        # Concatenate the additional information
        total_obs = torch.cat((map_obs_t, add_info_t), dim=1)

        # Pass through fully connected layers
        x = self.fc(total_obs)

        return self.policy(x), self.value(x)
    
