import numpy as np
import gymnasium as gym
from gymnasium import spaces


from pacman_game.map import *
from pacman_game.pacman import Pacman
from pacman_game.ghosts import *

from pacman_game.algorithms import NavigationAlgo
from pacman_game.constants import *

class PacmanEnv(gym.Env):
    def __init__(self, levels_paths, freq_change_level=1):
        super().__init__()

        # Check if level folder path is a list or a string
        if not isinstance(levels_paths, list):
            self.levels_paths = [levels_paths]
        else :
            self.levels_paths = levels_paths

        self.load_level(self.levels_paths[0])


        # Define the action space (up, down, left, right)
        self.action_space = spaces.MultiDiscrete(4)
        self.possible_actions = 4

        # Define the observation space
        # All the agents see the same thing : the level matrix (only the type of the cell not the tile index)
        self.observation_space = spaces.Box(low=0, high=12, shape=self.map.type_map.shape, dtype=np.uint8)

        # Instantiate the navigation algorithms
        self.navigation_algo = NavigationAlgo(self.map.type_map)

        self.freq_change_level = freq_change_level
        self.ep_before_change_level = freq_change_level

        self.max_steps = MAX_STEPS

    def load_level(self, level_csv_path:str):
        # Load the map
        self.current_level_name = level_csv_path
        self.map = Map(level_csv_path)

        # Parse the number of players and the initial positions ---------------
        self.agents: Pacman = []
        for _, spawn_position in enumerate(self.map.pacman_spawns):
            self.agents.append(Pacman(spawn_position))

        # Get the number of agents
        self.nb_agents = len(self.agents)
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


    def step(self, actions):

        self.navigation_algo.current_map = self.map.type_map

        # Initialize the rewards for all agents
        rewards = np.zeros(self.nb_agents)

        truncated = False
        info = {"ghosts_paths": {}}
        done = False

        # Handle the actions of the pacman agents
        for agent_index, action in enumerate(actions):

            # Select the agent
            agent = self.agents[agent_index]

            # Check if the agent is alive
            if not agent.alive:
                continue

            rewards[agent_index] += REWARDS["RW_LIVING"]

            # Decrease the superpower step left
            if agent.superpower_step_left > 0:
                agent.superpower_step_left -= 1

            # Apply the action
            candidate_position = agent.position + ACTION_MAP[action]

            # Check if the agent is turning back
            if agent.last_action != -1 and action == OPPOSITE_ACTION[agent.last_action]:
                rewards[agent_index] += REWARDS["RW_TURNING_BACK"]

            # Update the last direction
            agent.last_action = action

            # Restrain the candidate position to the map size (portal effect)
            candidate_position = (candidate_position + self.map.type_map.shape) % self.map.type_map.shape

            # Get the type of the candidate cell
            candidate_cell_type = self.map.type_map[tuple(candidate_position)]
            # Handle action/rewards based on the candidate cell type
            if candidate_cell_type in [WALL, GHOST_DOOR]:
                rewards[agent_index] += REWARDS["RW_NO_MOVE"]
                continue

            elif candidate_cell_type == DOOR :

                if agent.has_key:
                    rewards[agent_index] += REWARDS["RW_EMPTY"]
                else:
                    rewards[agent_index] += REWARDS["RW_NO_MOVE"]
                    continue

            elif candidate_cell_type == GUM:
                # Pick it up
                self.map.type_map[candidate_position[0], candidate_position[1]] = EMPTY
                agent.pacgum_eaten += 1
                self.nb_pacgum -= 1
                # Reward the agent
                rewards[agent_index] += REWARDS["RW_GUM"]
                agent.score += 10

            elif candidate_cell_type == SUPER_GUM:
                # Pick it up
                self.map.type_map[candidate_position[0], candidate_position[1]] = EMPTY
                # Set the superpower step left
                agent.superpower_step_left = 60
                # Reward the agent
                rewards[agent_index] += REWARDS["RW_SUPER_GUM"]
                agent.score += 50

            elif candidate_cell_type == FRUIT:
                # Pick it up
                self.map.type_map[candidate_position[0], candidate_position[1]] = EMPTY
                # Reward the agent
                rewards[agent_index] += REWARDS["RW_FRUIT"]
                agent.score += 100

            elif candidate_cell_type == KEY:
                # Pick it up
                self.map.type_map[candidate_position[0], candidate_position[1]] = EMPTY
                agent.has_key = True
                # Reward the agent
                rewards[agent_index] += REWARDS["RW_KEY"]

            elif candidate_cell_type == EMPTY:
                rewards[agent_index] += REWARDS["RW_EMPTY"]

            # Check for collision with other pacman (here pacman collides with pacman and cannot pass through)
            no_obstacles = True
            if self.nb_agents > 1:
                for index, other_agent in enumerate(self.agents):
                    if index != agent_index and other_agent.alive and np.all(other_agent.position == candidate_position):
                        # Don't move
                        no_obstacles = False
                        rewards[agent_index] += REWARDS["RW_NO_MOVE"] 

            # Move the agent
            if no_obstacles:
                agent.position = candidate_position

                # Update the exploration map of the agent (- reward for exploring already explored area)
                if agent.exploration_map[candidate_position[0], candidate_position[1]] == 0:
                    rewards[agent_index] += REWARDS["EXPLORE_REWARD"]
                else:
                    rewards[agent_index] += REWARDS["ALREADY_EXPLORED"] * agent.exploration_map[candidate_position[0], candidate_position[1]]
                
                agent.exploration_map[candidate_position[0], candidate_position[1]] += 1
            

            # Check for pacman collision with the ghosts (here pacman collides with ghost)
            for ghost in self.ghosts:
                # Check if pacman is colliding with the ghost
                if np.all(ghost.position == candidate_position):
                    # Check if the agent is in super pacman mode
                    if agent.superpower_step_left > 0:
                        # Kill the ghost
                        ghost.reset()
                        # Reward the agent
                        rewards[agent_index] += REWARDS["RW_EATING_GHOST"]

                        agent.score += 200
                    else:
                        # Kill the pacman
                        agent.alive = False
                        self.alive_agents -= 1
                        # Reward the agent
                        rewards[agent_index] += REWARDS["RW_DYING_TO_GHOST"]


        for ghost in self.ghosts:
            # If the ghost is not free then skip its turn
            if not ghost.is_free:
                continue

            # Choose the direction
            ghost_action, ghost_path = ghost.choose_direction(self.agents, self.navigation_algo, self.ghosts)
            candidate_ghost_position = ghost.position + ACTION_MAP[ghost_action]

            if ghost_path:
                info["ghosts_paths"][ghost.ghost_type] = ghost_path

            # Check if the candidate position is valid
            candidate_ghost_position = (candidate_ghost_position + self.map.type_map.shape) % self.map.type_map.shape
            # Check if the type of the cell
            cell_type = self.map.type_map[tuple(candidate_ghost_position)]

            # If the cell is a wall then the ghost can't move
            if cell_type == WALL:
                continue

            ghost_died = False
            # Check for ghost collision with pacman (here the ghost collides with pacman)
            for agent in self.agents:
                # Check if the ghost is colliding with pacman
                if np.all(agent.position == candidate_ghost_position) and agent.alive:
                    # Check if the pacman agent is in super pacman mode
                    if agent.superpower_step_left > 0:
                        # Kill the ghost
                        ghost.reset()
                        # Reward the pacman agent
                        rewards[agent_index] += REWARDS["RW_EATING_GHOST"]
                        agent.score += 200
                        ghost_died = True
                    else:
                        # Kill the pacman
                        agent.alive = False
                        self.alive_agents -= 1
                        # Reward the pacman agent
                        rewards[agent_index] += REWARDS["RW_DYING_TO_GHOST"]

            # Move the ghost
            if not ghost_died:
                ghost.position = candidate_ghost_position

        # Increase the step count of the episode
        self.current_step += 1


        # Check if the max steps are reached
        if self.current_step >= self.max_steps :
            done = True
            truncated = True

            info = {
                "end_cause": "Episode truncated. Max steps reached"
            }

            for agent in self.agents:
                rewards[agent_index] +=  -1 * self.nb_pacgum

        # Check if all the agents are dead
        elif self.alive_agents <= 0:
            done = True

            info = {
                "end_cause": "Episode done. All the agents are dead"
            }


        # Check if all the pacgum are eaten
        elif self.nb_pacgum == 0:
            for agent in self.agents:
                rewards[agent_index] += REWARDS["RW_WINNING"]
            done = True

            info = {
                "end_cause": "Episode done. All the pacgum are eaten"
            }

        # Get the observations
        observations = self._get_observations()

        return observations, rewards, done, truncated, info

    def _get_observation_for_agent(self, agent_index):
        # Take the map
        map_copy = self.map.type_map.copy()

        # Put the pacman agent on the map
        for index, pacman_agent in enumerate(self.agents):
            if pacman_agent.alive :
                if index == agent_index:
                    map_copy[tuple(pacman_agent.position)] = PACMAN
                else :
                    map_copy[tuple(pacman_agent.position)] = PACMAN_RIVAL

        # Put the ghost agent on the map
        for ghost in self.ghosts:
            map_copy[tuple(ghost.position)] = ghost.ghost_type

        return map_copy


    def _get_observations(self):
        # Get observations for all agents
        observations = [[] for _ in range(self.nb_agents)]
        for i in range(self.nb_agents) :
            observation = self._get_observation_for_agent(i)

            # Append the matrix of the map to the observation
            observations[i].append(observation)

            # Append additional information to the observation

            # Get the score of the other agents, there is either 1 or 2 agents
            score_other_agent = 0

            if self.nb_agents > 1:
                score_other_agent = self.agents[1 - i].score

            observations[i].append(
                [int(self.agents[i].position[0]), int(self.agents[i].position[1]), # Position x, y of the agent
                 int(self.agents[i].last_action), # Last action of the agent
                 int(self.agents[i].superpower_step_left),  # Superpower step left
                 int(score_other_agent), int(self.agents[i].score), # Score of the other agent and this agent
                 int(self.agents[i].has_key) # Has key
                 ]
            )

        return observations

    def reset(self, seed=None):

        info = {}

        self.current_step = 0

        # If the level folder path is a list then choose a random level
        if len(self.levels_paths) > 1 and self.ep_before_change_level <= 0:
            level_index = np.random.randint(0, len(self.levels_paths))
            self.load_level(self.levels_paths[level_index])
            self.ep_before_change_level = self.freq_change_level
        
        self.ep_before_change_level -= 1
        # Reset the map
        self.map.reset()

        # Reset the agents
        for agent in self.agents:
            agent.reset()
        self.alive_agents = self.nb_agents

        for ghost in self.ghosts:
            ghost.reset()

        self.nb_pacgum = self.nb_pacgum_start

        self.navigation_algo.current_map = self.map.type_map

        return self._get_observations(), info


    def render(self, mode='rgb_array', infos=None):
        return self.map.render(mode, self.agents, self.ghosts, infos)