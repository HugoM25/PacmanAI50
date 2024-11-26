import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box


from pacman_game.map import *
from pacman_game.pacman import Pacman
from pacman_game.ghosts import *

MAX_STEPS = 1000

REWARDS = {
    "RW_GUM": 10,
    "RW_SUPER_GUM": 10,
    "RW_EMPTY": 0,
    "RW_NO_MOVE": 0,
    "RW_DYING_TO_GHOST": -500,
    "RW_EATING_GHOST": 50,
    "RW_WINNING": 100
}

ACTION_MAP = {
    0: np.array([-1, 0]),  # Up
    1: np.array([1, 0]),   # Down
    2: np.array([0, -1]),  # Left
    3: np.array([0, 1]),   # Right
}

class PacmanEnv(gym.Env):

    metadata = {"render_modes": ["human"], "name": "pacman-env-v0"}

    def __init__(self, level_folder_path:str, flatten_observation:bool = False, render_mode="human"):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = MAX_STEPS

        # Load the map
        self.map = Map(level_folder_path)

        # Parse the number of players and the initial positions ---------------
        self.possible_agents: Pacman = []
        for _, spawn_position in enumerate(self.map.pacman_spawns):
            self.possible_agents.append(Pacman(spawn_position))

        # Get the number of agents
        self.nb_agents = len(self.possible_agents)
        self.alive_agents = self.nb_agents

        self.current_step = 0

        # Parse the ghosts initial positions -----------------------------------
        self.ghosts: Ghost = []
        inky_spawn, blinky_spawn, pinky_spawn, clyde_spawn = self.map.ghosts_spawns

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

        # Get the number of pacgum to eat
        self.nb_pacgum_start = np.sum(self.map.type_map == GUM)
        self.nb_pacgum = self.nb_pacgum_start

        # ---------------------------------------------------------------------
        # Define the action space (up, down, left, right)
        self.action_spaces = {
            agent: Discrete(4) for agent in self.possible_agents
        }

        self.observation_spaces = {
            agent: Box(low=0, high=12, shape=self.map.type_map.shape, dtype=np.uint8) for agent in self.possible_agents
        }

        self.flatten_observation = flatten_observation



    def step(self, actions):

        # Initialize the rewards for all agents
        rewards = {agent: 0 for agent in self.agents}
        # Initialize the done flag
        terminations = {agent: False for agent in self.agents}
        # Initialize the truncated flag
        truncations = {agent: False for agent in self.agents}
        # Initialize the info dict
        infos = {agent: {} for agent in self.agents}


        # Increase the step count of the episode
        self.current_step += 1

        # Handle the actions of the pacman agents
        for agent_index, action in enumerate(actions):

            # Select the agent
            agent = self.possible_agents[agent_index]

            # Check if the agent is alive
            if not agent.alive:
                continue

            # Decrease the superpower step left
            if agent.superpower_step_left > 0:
                agent.superpower_step_left -= 1

            # Apply the action
            candidate_position = agent.position + ACTION_MAP[action]

            # Restrain the candidate position to the map size (portal effect)
            candidate_position = (candidate_position + self.map.type_map.shape) % self.map.type_map.shape

            # Get the type of the candidate cell
            candidate_cell_type = self.map.type_map[tuple(candidate_position)]

            # Handle action/rewards based on the candidate cell type
            if candidate_cell_type in [WALL, PACMAN_RIVAL, GHOST_DOOR]:
                rewards[agent] = REWARDS["RW_NO_MOVE"]
                continue

            elif candidate_cell_type == GUM:
                # Pick it up
                self.map.type_map[candidate_position[0], candidate_position[1]] = EMPTY
                agent.pacgum_eaten += 1
                self.nb_pacgum -= 1
                # Reward the agent
                rewards[agent] = REWARDS["RW_GUM"]

            elif candidate_cell_type == SUPER_GUM:
                # Pick it up
                self.map.type_map[candidate_position[0], candidate_position[1]] = EMPTY
                # Set the superpower step left
                agent.superpower_step_left = 60
                # Reward the agent
                rewards[agent] = REWARDS["RW_SUPER_GUM"]

            elif candidate_cell_type == EMPTY:
                rewards[agent] = REWARDS["RW_EMPTY"]

            # Check for pacman collision with the ghosts (here pacman collides with ghost)
            for ghost in self.ghosts:
                # Check if pacman is colliding with the ghost
                if np.all(ghost.position == candidate_position):
                    # Check if the agent is in super pacman mode
                    if agent.superpower_step_left > 0:
                        # Kill the ghost
                        ghost.reset()
                        # Reward the agent
                        rewards[agent] = REWARDS["RW_EATING_GHOST"]
                    else:
                        # Kill the pacman
                        agent.alive = False
                        self.alive_agents -= 1
                        # Reward the agent
                        rewards[agent] = REWARDS["RW_DYING_TO_GHOST"]

            # Move the agent
            agent.position = candidate_position


        for ghost in self.ghosts:
            # If the ghost is not free then skip its turn
            if not ghost.is_free:
                continue

            pacman_positions = [agent.position for agent in self.possible_agents if agent.alive]
            # Choose the direction
            ghost_action = ghost.choose_direction(self.map.type_map, pacman_positions)
            candidate_ghost_position = ghost.position + ACTION_MAP[ghost_action]

            # Check if the candidate position is valid
            candidate_ghost_position = (candidate_ghost_position + self.map.type_map.shape) % self.map.type_map.shape

            # Check if the type of the cell
            cell_type = self.map.type_map[tuple(candidate_ghost_position)]

            # If the cell is a wall then the ghost can't move
            if cell_type == WALL:
                continue

            # Check for ghost collision with pacman (here the ghost collides with pacman)
            for agent in self.possible_agents:
                # Check if the ghost is colliding with pacman
                if np.all(agent.position == candidate_ghost_position) and agent.alive:
                    # Check if the pacman agent is in super pacman mode
                    if agent.superpower_step_left > 0:
                        # Kill the ghost
                        ghost.reset()
                        # Reward the pacman agent
                        rewards[agent] = REWARDS["RW_EATING_GHOST"]
                    else:
                        # Kill the pacman
                        agent.alive = False
                        self.alive_agents -= 1
                        # Reward the pacman agent
                        rewards[agent] = REWARDS["RW_DYING_TO_GHOST"]

            # Move the ghost
            ghost.position = candidate_ghost_position

        # Check if the max steps are reached
        if self.current_step >= self.max_steps :
            # For all the agents not dead or winning, give a negative reward for taking too long
            for agent in self.possible_agents:
                if agent.alive :
                    rewards[agent] += REWARDS["RW_NO_MOVE"]


        # Check if all the agents are dead
        elif self.alive_agents <= 0:
            done = True

            info = {
                "end_cause": "Episode done. All the agents are dead"
            }

        # Check if all the pacgum are eaten
        elif self.nb_pacgum == 0:
            # Give rewards based on steps left to the agents alive
            for agent in self.possible_agents:
                rewards[agent] = REWARDS["RW_WINNING"] / self.current_step

            done = True

            info = {
                "end_cause": "Episode done. All the pacgum are eaten"
            }

        # Get the observations
        observations = self._get_observations()

        return observations, rewards, terminations, truncations, infos


    def _get_observation_for_agent(self, agent_index: int) -> np.ndarray:
        '''
        Get the observation for a specific agent
        The observation is a 2D array representing the map with the agent on it -
        The agent is represented by a pacman value and the others by a rival pacman value
        :param agent_index: the index of the agent
        @return: the observation
        '''
        map_copy = self.map.type_map.copy()

        # Put the pacman agent on the map
        for index, pacman_agent in enumerate(self.possible_agents):
            if pacman_agent.alive :
                if index == agent_index:
                    map_copy[tuple(pacman_agent.position)] = PACMAN
                else :
                    map_copy[tuple(pacman_agent.position)] = PACMAN_RIVAL

        # Put the ghost agent on the map
        for ghost in self.ghosts:
            map_copy[tuple(ghost.position)] = ghost.ghost_type

        return map_copy.flatten() if self.flatten_observation else map_copy


    def _get_observations(self) -> list[np.ndarray]:
        '''
        Get observations for all agents
        @return: list of observations
        '''

        observations = []
        for i in range(self.nb_agents) :
            observation = self._get_observation_for_agent(i)
            observations.append(observation)
        return observations

    def reset(self):
        '''
        Reset the environment (and setup the env to make sure render(), step(), etc. work correctly)
        '''

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: self._get_observation_for_agent(index) for index, agent in enumerate(self.agents)}

        self.current_step = 0

        # Reset the map
        self.map.reset()

        # Reset the ghosts
        for ghost in self.ghosts:
            ghost.reset()

        # Keep track of the number of pacgum to eat
        self.nb_pacgum = self.nb_pacgum_start

        return self.observations, self.infos


    def render(self, mode='rgb_array'):
        return self.map.render(mode, self.possible_agents, self.ghosts)
