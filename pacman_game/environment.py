import numpy as np 
import gymnasium as gym
from gymnasium import spaces


from pacman_game.map import *
from pacman_game.pacman import Pacman

MAX_STEPS = 1000

class PacEnv(gym.Env): 
    def __init__(self, level_folder_path:str):
        super().__init__()

        # Load the map
        self.map = Map(level_folder_path)
        
        # Parse the number of players and the initial positions ---------------
        self.agents = []
        for _, spawn_position in enumerate(self.map.get_pacman_agents_positions()):
            self.agents.append(Pacman(spawn_position))
        
        # Get the number of agents
        self.nb_agents = len(self.agents)

        # Parse the ghosts initial positions -----------------------------------
        # TODO : Parse the ghosts initial positions

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

            if not agent.alive:
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
            else :
                continue
            
            # Ensure the candidate position is valid
            if candidate_position[0] < 0 or candidate_position[0] >= self.map.type_map.shape[0] or candidate_position[1] < 0 or candidate_position[1] >= self.map.type_map.shape[1] :
                # Candidate position is out of the map, do nothing
                rewards[agent_index] = -10
            elif self.map.type_map[candidate_position[0], candidate_position[1]] in [WALL, PACMAN_RIVAL, GHOST_DOOR]: 
                # Candidate position is a wall or another pacman agent, do nothing
                rewards[agent_index] = -10
            else :   
                # Candidate position is valid check the cell type

                candidate_cell_type = self.map.type_map[candidate_position[0], candidate_position[1]]
                # Handle rewards based on cell type
                if candidate_cell_type == GUM:
                    rewards[agent_index] = 2
                    agent.pacgum_eaten += 1
                elif candidate_cell_type == SUPER_GUM:
                    rewards[agent_index] = 10
                elif candidate_cell_type in [GHOST_INKY, GHOST_BLINKY, GHOST_PINKY, GHOST_CLYDE]:
                    self.agents[agent_index].alive = False
                    rewards[agent_index] = -10
                elif candidate_cell_type == EMPTY:
                    rewards[agent_index] = -0.05

                
                # Move the agent
                self.map.type_map[agent.position[0], agent.position[1]] = EMPTY
                agent.position = candidate_position
                self.map.type_map[agent.position[0], agent.position[1]] = PACMAN

        # Handle the ghosts actions
        # TODO : Handle the ghosts actions

        self.current_step += 1
        # Check if the episode is done
        if self.current_step >= self.max_steps or np.sum(self.map.type_map == PACMAN) == 0 or np.sum(self.map.type_map == GUM) == 0:
            done = True

        observations = self._get_observations()

        info["gum_collected"] = self.agents[0].pacgum_eaten      
        info["max_gum"] = np.sum(self.map.type_map == GUM)

        return observations, rewards, done, truncated, info
    
    def _get_observation_for_agent(self, agent_index):
        # Take the map 
        map = self.map.type_map.copy()
        # Replace the pacman by the number 11
        map[map == PACMAN] = PACMAN_RIVAL

        # Add the agent to the map
        agent = self.agents[agent_index]
        map[agent.position[0], agent.position[1]] = PACMAN

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

        print("Reset done")

        return self._get_observations(), info
        

    def render(self, mode='rgb_array'):
        return self.map.render(mode)
