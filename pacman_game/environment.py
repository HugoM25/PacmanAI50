import numpy as np
import gymnasium as gym
from gymnasium import spaces


from pacman_game.map import *
from pacman_game.pacman import Pacman
from pacman_game.ghosts import *

MAX_STEPS = 5000

class PacEnv(gym.Env):
    def __init__(self, level_folder_path:str):
        super().__init__()

        # Load the map
        self.map = Map(level_folder_path)

        # Parse the number of players and the initial positions ---------------
        self.agents: Pacman = []
        for _, spawn_position in enumerate(self.map.pacman_spawns):
            self.agents.append(Pacman(spawn_position))

        # Get the number of agents
        self.nb_agents = len(self.agents)
        self.alive_agents = self.nb_agents

        # Parse the ghosts initial positions -----------------------------------
        # TODO : Parse the ghosts initial positions
        self.ghosts: Ghost = []
        blinky_spawn, pinky_spawn, inky_spawn, clyde_spawn = self.map.ghosts_spawns

        if blinky_spawn :
            self.ghosts.append(Blinky(blinky_spawn))
        if pinky_spawn :
            self.ghosts.append(Pinky(pinky_spawn))
        if inky_spawn :
            self.ghosts.append(Inky(inky_spawn))
        if clyde_spawn :
            self.ghosts.append(Clyde(clyde_spawn))

        # Get the number of ghosts
        self.nb_ghosts = len(self.ghosts)

        # ---------------------------------------------------------------------
        # Define the action space (up, down, left, right)
        self.action_space = spaces.MultiDiscrete(4)

        # Define the observation space
        # All the agents see the same thing : the level matrix (only the type of the cell not the tile index)
        self.observation_space = spaces.Box(low=0, high=12, shape=self.map.type_map.shape, dtype=np.uint8)


    def step(self, actions):

        # Initialize the rewards for all agents
        rewards = [0] * self.nb_agents

        truncated = False
        info = {}
        done = False

        # Handle the actions of the agents
        for agent_index, action in enumerate(actions):
            # Select the agent
            agent = self.agents[agent_index]

            if not agent.is_alive:
                continue

            # Apply the action
            if action == 0:
                candidate_position = agent.position + np.array([-1, 0])
            elif action == 1:
                candidate_position = agent.position + np.array([1, 0])
            elif action == 2:
                candidate_position = agent.position + np.array([0, -1])
            elif action == 3:
                candidate_position = agent.position + np.array([0, 1])

            # Ensure the candidate position is valid
            if candidate_position[0] < 0 or candidate_position[0] >= self.map.type_map.shape[0] or candidate_position[1] < 0 or candidate_position[1] >= self.map.type_map.shape[1] :
                # Candidate position is out of the map, do nothing
                rewards[agent_index] = -1
            elif self.map.type_map[candidate_position[0], candidate_position[1]] in [WALL, PACMAN_RIVAL, GHOST_DOOR]:
                # Candidate position is a wall or another pacman agent, do nothing
                rewards[agent_index] = -1
            else :
                # Candidate position is valid check the cell type

                candidate_cell_type = self.map.type_map[candidate_position[0], candidate_position[1]]
                # Handle rewards based on cell type
                if candidate_cell_type == GUM:
                    rewards[agent_index] = 2
                    agent.pacgum_eaten += 1
                    self.map.type_map[agent.position[0], agent.position[1]] = EMPTY

                elif candidate_cell_type == SUPER_GUM:
                    rewards[agent_index] = 10
                    self.map.type_map[agent.position[0], agent.position[1]] = EMPTY

                elif candidate_cell_type == EMPTY:
                    rewards[agent_index] = -1

                elif candidate_cell_type in [GHOST_BLINKY, GHOST_PINKY, GHOST_INKY, GHOST_CLYDE]:
                    # The agent is dead
                    agent.is_alive = False
                    rewards[agent_index] = -10
                    self.alive_agents -= 1

                # Move the agent
                agent.position = candidate_position

        # Handle the ghosts actions
        # TODO : Handle ALL the ghosts actions

        for ghost in self.ghosts:
            # Maybe the ghost is locked then it can't move
            if not ghost.is_free:
                continue

            # Choose the direction
            direction = ghost.choose_direction(self.map.type_map)

            if direction == 0:
                candidate_ghost_position = ghost.position + np.array([-1, 0])
            elif direction == 1:
                candidate_ghost_position = ghost.position + np.array([1, 0])
            elif direction == 2:
                candidate_ghost_position = ghost.position + np.array([0, -1])
            elif direction == 3:
                candidate_ghost_position = ghost.position + np.array([0, 1])
            else :
                continue

            if candidate_ghost_position[0] < 0 or candidate_ghost_position[0] >= self.map.type_map.shape[0] or candidate_ghost_position[1] < 0 or candidate_ghost_position[1] >= self.map.type_map.shape[1] :
                continue

            cell_type = self.map.type_map[candidate_ghost_position[0], candidate_ghost_position[1]]

            if cell_type == WALL:
                continue

            # If there is a pacman agent on the cell
            for agent in self.agents:
                if np.all(agent.position == candidate_ghost_position):
                    agent.is_alive = False
                    rewards[agent_index] = -10
                    self.alive_agents -= 1

            ghost.position = candidate_ghost_position


        self.current_step += 1
        # Check if the episode is done
        if self.current_step >= self.max_steps or np.sum(self.map.type_map == GUM) == 0 or self.alive_agents == 0:
            done = True

        observations = self._get_observations()

        info["gum_collected"] = self.agents[0].pacgum_eaten
        info["max_gum"] = np.sum(self.map.type_map == GUM)

        return observations, rewards, done, truncated, info

    def _get_observation_for_agent(self, agent_index):
        # Take the map
        map = self.map.type_map.copy()

        # Put the pacman agent on the map
        for index, pacman_agent in enumerate(self.agents):
            if index == agent_index:
                map[pacman_agent.position[0], pacman_agent.position[1]] = PACMAN
            else :
                map[pacman_agent.position[0], pacman_agent.position[1]] = PACMAN_RIVAL

        # Put the ghost agent on the map
        for ghost in self.ghosts:
            map[ghost.position[0], ghost.position[1]] = ghost.ghost_type

        return map.flatten()

    def _get_observations(self):
        # Get observations for all agents
        observations = []
        for i in range(self.nb_agents) :
            observation = self._get_observation_for_agent(i)
            observations.append(observation)
        return observations

    def reset(self, seed=None):

        info = {}

        self.current_step = 0
        self.max_steps = MAX_STEPS

        # Reset the map
        self.map.reset()

        # Reset the agents
        for agent in self.agents:
            agent.reset()
        self.alive_agents = self.nb_agents

        for ghost in self.ghosts:
            ghost.reset()
        print("Reset done")

        return self._get_observations(), info


    def render(self, mode='rgb_array'):
        return self.map.render(mode, self.agents, self.ghosts)