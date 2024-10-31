import numpy as np
import csv 
import json 
import cv2


EMPTY = 0
GUM = 1
SUPER_GUM = 2
GHOST_INKY = 3
GHOST_BLINKY = 4
GHOST_PINKY = 5
GHOST_CLYDE = 6
PACMAN = 7
FRUIT = 8
WALL = 9
GHOST_DOOR = 10
PACMAN_RIVAL = 11

class Map :

    def __init__(self, level_folder_path=None):

        # Load the level from a CSV file
        self.tile_map = self.load_level_from_csv(level_folder_path + "level_data.csv")

        # Load the json file to get the type of each cell
        with open(level_folder_path + "info.json", "r") as f:
            self.level_json = json.load(f)

        # Load the tileset image from the level folder
        self.tileset = cv2.imread(level_folder_path + "tileset.png")
        # Separate the tileset into individual tiles
        self.tiles = []
        self.tile_size = self.level_json["tileset"]['tile_size']
        for i in range(self.tileset.shape[0] // self.tile_size):
            for j in range(self.tileset.shape[1] // self.tile_size):
                self.tiles.append(self.tileset[i*self.tile_size:(i+1)*self.tile_size, j*self.tile_size:(j+1)*self.tile_size])
        
        # Convert tile_map to type_map
        tile_id_to_type = {tile['id']: int(tile['type']) for tile in self.level_json['tileset']['tiles']}
        self.type_map = np.zeros_like(self.tile_map, dtype=np.uint8)
        for i in range(self.tile_map.shape[0]):
            for j in range(self.tile_map.shape[1]):
                self.type_map[i, j] = tile_id_to_type[self.tile_map[i, j]]
        
        # Save the initial maps
        self.initial_tile_map = self.tile_map.copy()
        self.initial_type_map = self.type_map.copy()


        # Render optimizations -----------------------------------------------
        # Create a "background" image for the map where no ghosts/pacman/gums are drawn
        self.background = np.zeros((self.tile_map.shape[0] * self.tile_size, self.tile_map.shape[1] * self.tile_size, 3), dtype=np.uint8)
        for i in range(self.type_map.shape[0]):
            for j in range(self.type_map.shape[1]):
                tile_t = self.type_map[i, j]
                if tile_t not in [WALL,GHOST_DOOR,EMPTY] : 
                    continue

                tile_i = self.tiles[self.tile_map[i, j]]
                self.background[i*self.tile_size:(i+1)*self.tile_size, j*self.tile_size:(j+1)*self.tile_size] = tile_i


    def load_level_from_csv(self, level_file_path: str):
        """
        Load a level from a CSV file.
        :param file_path: The path to the CSV file.
        :return: The level as a 2D numpy array of integers.
        """
        level = []
        with open(level_file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                level.append([int(cell) for cell in row])
        return np.array(level)
    
    def get_pacman_agents_positions(self):
        '''
        Get the positions of the pacman agents. 
        :return: A list of tuples containing the positions of the pacman agents.
        '''
        self.pacman_agents_spawns = []
        # The agents are represented by the numbers 7 on the type_map
        self.start_pos = np.where(self.type_map == 7)
        self.start_pos = list(zip(self.start_pos[0], self.start_pos[1]))

        return self.start_pos
    
    def get_ghost_agents_spawns(self):
        '''
        Get the positions of the ghost agents. 
        :return: A list of tuples containing the positions of the ghost agents.
        '''
    
    def reset(self):
        self.tile_map = self.initial_tile_map.copy()
        self.type_map = self.initial_type_map.copy()

    def render(self, mode: str):
        if mode == "rgb_array":
            return self._render_as_rgb_array()

        
    def _render_as_rgb_array(self): 
        '''
        Render the map as an RGB array.
        :return: The RGB array of the map as a numpy array
        '''
        map_img = self.background.copy()

        # Add the pacman agents to the image
        pacman_agents_positions = self.get_pacman_agents_positions()
        for agent_position in pacman_agents_positions:
            map_img[agent_position[0]*self.tile_size:(agent_position[0]+1)*self.tile_size, agent_position[1]*self.tile_size:(agent_position[1]+1)*self.tile_size] = self.tiles[6]

        for i in range(self.type_map.shape[0]):
            for j in range(self.type_map.shape[1]):
                tile_t = self.type_map[i, j]
                # Add the gum to the image
                if tile_t == GUM:
                    map_img[i*self.tile_size:(i+1)*self.tile_size, j*self.tile_size:(j+1)*self.tile_size] = self.tiles[self.tile_map[i, j]]
                # Add the super gum to the image
                if tile_t == SUPER_GUM:
                    map_img[i*self.tile_size:(i+1)*self.tile_size, j*self.tile_size:(j+1)*self.tile_size] = self.tiles[self.tile_map[i, j]]

        # Scale the image up
        map_img = cv2.resize(map_img, (map_img.shape[1]*2, map_img.shape[0]*2), interpolation=cv2.INTER_NEAREST)

        return map_img

    
        
    
