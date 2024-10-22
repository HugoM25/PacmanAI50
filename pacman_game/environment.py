import numpy as np
import os
import json

import gymnasium as gym
from gymnasium import spaces

from pacman_game.utils import load_level_from_csv
from pacman_game.pacman import Pacman

class PacmanEnv(gym.Env):
    def __init__(self, level_path=None) :
        super().__init__()

        # Load the level from a CSV file
        if level_path is None:
            # Stop the program if the level path is not provided
            raise ValueError("The level path must be provided.")

        self.level_tiles = load_level_from_csv(level_path)

        # Load the json file to get the type of each cell
        with open(os.path.join(os.path.dirname(level_path), "res/tileset_pacman1.json"), "r") as f:
            self.level_json = json.load(f)

        # Create a dictionary to map tile IDs to their types
        tile_id_to_type = {tile['id']: int(tile['type']) for tile in self.level_json['tileset']['tiles']}
        # Create a new matrix for the level with the same shape as the level tiles
        self.level = np.zeros_like(self.level_tiles, dtype=np.uint8)
        # Map the tile IDs to their types
        for i in range(self.level_tiles.shape[0]):
            for j in range(self.level_tiles.shape[1]):
                self.level[i, j] = tile_id_to_type[self.level_tiles[i, j]]

        # Save the initial level
        self.initial_level = self.level.copy()
        self.initial_level_tiles = self.level_tiles.copy()

        # Parse the number of players and the initial positions ---------------
        self.agents = []
        # The agents are represented by the numbers 6
        self.start_pos = np.where(self.level == 7)
        self.start_pos = list(zip(self.start_pos[0], self.start_pos[1]))
        for i, pos in enumerate(self.start_pos):
            self.agents.append((Pacman(pos)))

        self.nb_agents = len(self.agents)
        self.alive_agents = self.nb_agents

        # ---------------------------------------------------------------------
        # Define the action space (up, down, left, right)
        self.action_space = spaces.MultiDiscrete([5] * self.nb_agents)

        # Define the observation space
        # All the agents see the same thing : the level matrix (only the type of the cell not the tile index)
        self.observation_space = spaces.Box(low=0, high=11, shape=self.level.shape, dtype=np.uint8)

        self.current_step = 0
        self.max_steps = 11

        self.render()

    def step(self, actions):
        self.current_step += 1
        rewards = [0] * self.nb_agents  # Initialize rewards for all agents
        done = False
        truncated = False
        info = {}

        # Handle the actions of the agents
        for i in range(self.nb_agents):
            print(f"Agent {i} wants to move", end="")
            if actions[i] == 0:
                candidate_pos = self.agents[i].position + np.array([-1, 0])  # Move up
                print(" up", end="")
            elif actions[i] == 1:
                candidate_pos = self.agents[i].position + np.array([1, 0])   # Move down
                print(" down", end="")
            elif actions[i] == 2:
                candidate_pos = self.agents[i].position + np.array([0, -1])  # Move left
                print(" left", end="")
            elif actions[i] == 3:
                candidate_pos = self.agents[i].position + np.array([0, 1])   # Move right
                print(" right", end="")
            else:
                print(" Agent does nothing")
                continue

            print(f" to {candidate_pos}")
            cell_type = self.level[candidate_pos[0], candidate_pos[1]]

            # Check cell type and handle movement logic...
            if cell_type in [9, 7]:  # Wall or another pacman
                print("Agent stays in the same position")
                continue

            # Handle rewards based on cell type
            if cell_type == 7:  # Pacgum
                rewards[i] += 1  # Increment the specific agent's reward
                self.agents[i].pacgum_eaten += 1
            elif cell_type == 8:  # Ghost
                rewards[i] -= 1  # Decrement the specific agent's reward
                self.agents[i].alive = False
                self.alive_agents -= 1

            # Move the agent to the new position
            self.level[self.agents[i].position[0], self.agents[i].position[1]] = 11
            self.agents[i].position = candidate_pos
            self.level[self.agents[i].position[0], self.agents[i].position[1]] = 7

        # Check if the episode is done
        if self.current_step >= self.max_steps or self.alive_agents == 0 or np.sum(self.level == 7) == 0:
            done = True

        # Return the observation, individual rewards, done, truncated, and info
        return self.level.flatten(), rewards, done, truncated, info

    def reset(self, seed=None) :
        self.current_step = 0
        self.alive_agents = self.nb_agents
        self.state = self.initial_level.copy()
        return self.level, {}

    def render(self) :
        '''
        Render the environment using the level matrix
        Display the level matrix as a grid of colored cells
        '''
        sp = " "
        print(f"Step: {self.current_step}")
        for i in range(self.level.shape[0]):
            for j in range(self.level.shape[1]):
                cell_type = self.level[i, j]

                if cell_type == 1:
                    # Pacgum
                    print(".", end=sp)
                elif cell_type == 2:
                    # Super pacgum
                    print("o", end=sp)
                elif cell_type == 3:
                    # Ghost (inky)
                    print("I", end=sp)
                elif cell_type == 4:
                    # Ghost (pinky)
                    print("P", end=sp)
                elif cell_type == 5:
                    # Ghost (blinky)
                    print("B", end=sp)
                elif cell_type == 6:
                    # Ghost (clyde)
                    print("L", end=sp)
                elif cell_type == 7:
                    # Pacman
                    print("C", end=sp)
                elif cell_type == 8:
                    # Fruit
                    print("F", end=sp)
                elif cell_type == 9:
                    # Wall
                    print("#", end=sp)
                elif cell_type == 10:
                    # Ghost door
                    print("-", end=sp)
                elif cell_type == 11:
                    # Empty cell
                    print(" ", end=sp)
                else:
                    print(" ", end=sp)
            # Print a new line at the end of each row
            print()


