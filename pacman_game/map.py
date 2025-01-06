import numpy as np
import csv
import json
import cv2

from pacman_game.constants import *

class Map :

    def __init__(self, level_csv_path=None):

        # Load the level from a CSV file
        self.tile_map = self.load_level_from_csv(level_csv_path)

        # Get the res folder path from the level_csv_path
        level_csv_path.split("/")
        # remove last two elements
        res_folder_path = "/".join(level_csv_path.split("/")[:-2]) + "/"

        # Load the json file to get the type of each cell
        with open( res_folder_path + "assets/info.json", "r") as f:
            self.level_json = json.load(f)

        # Load the tileset image from the level folder
        self.tileset = cv2.imread(res_folder_path + "assets/tileset.png", cv2.IMREAD_UNCHANGED)
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

        # Save the spawns of the agents
        self.pacman_spawns = self.get_pacman_agents_positions()
        self.ghosts_spawns = self.get_ghost_agents_positions()

        # Remove the pacman and the ghosts from the map
        for pacman_spawn in self.pacman_spawns:
            self.type_map[pacman_spawn[0], pacman_spawn[1]] = EMPTY
            self.tile_map[pacman_spawn[0], pacman_spawn[1]] = 59
        for ghost_spawn in self.ghosts_spawns:
            if ghost_spawn is not None:
                self.type_map[ghost_spawn[0], ghost_spawn[1]] = EMPTY
                self.tile_map[ghost_spawn[0], ghost_spawn[1]] = 59

        # Save the initial maps
        self.initial_tile_map = self.tile_map.copy()
        self.initial_type_map = self.type_map.copy()

        # Render optimizations -----------------------------------------------
        # Create a "background" image for the map where no ghosts/pacman/gums are drawn
        self.background = np.zeros((self.tile_map.shape[0] * self.tile_size, self.tile_map.shape[1] * self.tile_size, 4), dtype=np.uint8)
        for i in range(self.type_map.shape[0]):
            for j in range(self.type_map.shape[1]):
                tile_t = self.type_map[i, j]
                if tile_t not in [WALL,GHOST_DOOR,EMPTY,DOOR] :
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

    def get_ghost_agents_positions(self):
        '''
        Get the positions of the ghost agents.
        :return: A list of tuples containing the positions of the ghost agents.
        '''
        self.ghosts_agents_spawns = []
        # The agents are represented by the numbers 3, 4, 5, 6 on the type_map

        self.inky_spawn = np.where(self.type_map == 3)
        if len(self.inky_spawn[0]) == 0:
            self.inky_spawn = None
        else:
            self.inky_spawn = list(zip(self.inky_spawn[0], self.inky_spawn[1]))[0]

        self.blinky_spawn = np.where(self.type_map == 4)
        if len(self.blinky_spawn[0]) == 0:
            self.blinky_spawn = None
        else :
            self.blinky_spawn = list(zip(self.blinky_spawn[0], self.blinky_spawn[1]))[0]

        self.pinky_spawn = np.where(self.type_map == 5)
        if len(self.pinky_spawn[0]) == 0:
            self.pinky_spawn = None
        else:
            self.pinky_spawn = list(zip(self.pinky_spawn[0], self.pinky_spawn[1]))[0]

        self.clyde_spawn = np.where(self.type_map == 6)
        if len(self.clyde_spawn[0]) == 0:
            self.clyde_spawn = None
        else :
            self.clyde_spawn = list(zip(self.clyde_spawn[0], self.clyde_spawn[1]))[0]

        return self.inky_spawn, self.blinky_spawn, self.pinky_spawn, self.clyde_spawn

    def reset(self):
        self.tile_map = self.initial_tile_map.copy()
        self.type_map = self.initial_type_map.copy()

    def render(self, mode: str, pacman_agents, ghost_agents, infos=None):
        if mode == "rgb_array":
            return self._render_as_rgb_array(pacman_agents=pacman_agents, ghost_agents=ghost_agents, infos=infos)


    def _render_as_rgb_array(self, pacman_agents, ghost_agents, infos=None):
        '''
        Render the map as an RGB array.
        :return: The RGB array of the map as a numpy array
        '''
        font = cv2.FONT_HERSHEY_SIMPLEX
        map_img = self.background.copy()

        for i in range(self.type_map.shape[0]):
            for j in range(self.type_map.shape[1]):
                tile_t = self.type_map[i, j]
                # Add the gum to the image
                if tile_t == GUM:
                    map_img[i*self.tile_size:(i+1)*self.tile_size, j*self.tile_size:(j+1)*self.tile_size] = self.tiles[self.tile_map[i, j]]
                # Add the super gum to the image
                if tile_t == SUPER_GUM:
                    map_img[i*self.tile_size:(i+1)*self.tile_size, j*self.tile_size:(j+1)*self.tile_size] = self.tiles[self.tile_map[i, j]]
                # Add the fruit to the image
                if tile_t == FRUIT:
                    map_img[i*self.tile_size:(i+1)*self.tile_size, j*self.tile_size:(j+1)*self.tile_size] = self.tiles[self.tile_map[i, j]]
                # Add the key to the image
                if tile_t == KEY:
                    map_img[i*self.tile_size:(i+1)*self.tile_size, j*self.tile_size:(j+1)*self.tile_size] = self.tiles[self.tile_map[i, j]]

        # Add the ghosts to the image
        for ghost in ghost_agents:
            if not ghost.is_free:
                continue

            x, y = ghost.position
            tile_region = map_img[x*self.tile_size:(x+1)*self.tile_size, y*self.tile_size:(y+1)*self.tile_size, :3]
            ghost_tile = self.tiles[ghost.ghost_type-1][:, :, :3]
            alpha_ghost = self.tiles[ghost.ghost_type-1][:, :, 3] / 255.0
            for c in range(3):
                tile_region[:, :, c] = (alpha_ghost * ghost_tile[:, :, c] +
                            (1 - alpha_ghost) * tile_region[:, :, c])


            # # Write its x,y position on top of the ghost
            # cv2.putText(map_img, f"({x},{y})", (y*self.tile_size + 5, x*self.tile_size + 15), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        # Add the pacman agents to the image with transparency
        pacman_tile = self.tiles[6][:, :, :]

        for pacman in pacman_agents:
            if not pacman.alive:
                continue

            # Rotate the pacman image according to the last action

            # Base pacman is facing right
            pacman_tile_rotated = pacman_tile

            if pacman.last_action == 0:
                # Rotate the pacman image to face up
                pacman_tile_rotated = np.rot90(pacman_tile, 1)
            elif pacman.last_action == 1:
                # Mirror the pacman image to face down
                pacman_tile_rotated = np.flipud(pacman_tile)
            elif pacman.last_action == 2:
                pacman_tile_rotated = np.fliplr(pacman_tile)


            x, y = pacman.position

            # Add the pacman to the image
            map_img[x*self.tile_size:(x+1)*self.tile_size, y*self.tile_size:(y+1)*self.tile_size] = pacman_tile_rotated

        if infos is not None :
            # Show the path planned by the ghosts
            if 'ghosts_paths' in infos:
                        colors_paths = [(0, 0, 255), (255, 204, 0), (255, 102, 204), (0, 204, 255)]
                        for color_index, ghost_type in enumerate([GHOST_BLINKY, GHOST_INKY, GHOST_PINKY, GHOST_CLYDE]):
                            if ghost_type in infos['ghosts_paths'] and infos['ghosts_paths'][ghost_type] is not None:
                                path = infos['ghosts_paths'][ghost_type]
                                for i in range(len(path)-1):
                                    start_pos = path[i]
                                    end_pos = path[i + 1]
                                    start_x, start_y = start_pos
                                    end_x, end_y = end_pos
                                    cv2.line(map_img, (start_y*self.tile_size + self.tile_size//2, start_x*self.tile_size + self.tile_size//2),
                                            (end_y*self.tile_size + self.tile_size//2, end_x*self.tile_size + self.tile_size//2), colors_paths[color_index], 1)

        # Add padding to the image
        map_img = cv2.copyMakeBorder(map_img, 50, 50, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])

        # Scale the image up
        map_img = cv2.resize(map_img, (map_img.shape[1]*2, map_img.shape[0]*2), interpolation=cv2.INTER_NEAREST)

        # Add the score of the pacman agents to the image at the top left corner of the image
        for i, pacman in enumerate(pacman_agents):
            cv2.putText(map_img, f"J{i} score : {pacman.score}", (20, 20 + i*20), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


        if infos is not None:
            # Add the episode number to the image at the top right corner of the image
            if 'episode' in infos and infos['episode'] is not None:
                cv2.putText(map_img, f"Episode : {infos['episode']}", (map_img.shape[1] - 200, 20), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            # Add the step number to the image at the top right corner of the image
            if 'step' in infos and infos['step'] is not None:
                cv2.putText(map_img, f"Step : {infos['step']}", (map_img.shape[1] - 200, 40), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            if 'total_steps' in infos and infos['total_steps'] is not None:
                # Simplify the total steps by writing 1k, 1M, 1B, 1T instead of 1000, 1000000, 1000000000, 1000000000000
                total_steps = infos['total_steps']
                if total_steps >= 1000000000000:
                    total_steps = f"{total_steps // 1000000000000}T"
                elif total_steps >= 1000000000:
                    total_steps = f"{total_steps // 1000000000}B"
                elif total_steps >= 1000000:
                    total_steps = f"{total_steps // 1000000}M"
                elif total_steps >= 1000:
                    total_steps = f"{total_steps // 1000}k"
                cv2.putText(map_img, f"Total steps : {total_steps}", (map_img.shape[1] - 200, 60), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                                
            # Add the probabilities of the pacman agents taking an action at the bottom (as an histogram)
            if 'probabilities_moves' in infos and infos['probabilities_moves'] is not None:
                for x, agent_moves_prob in enumerate(infos['probabilities_moves']):
                    for i, probs in enumerate(agent_moves_prob):
                        for j, prob in enumerate(probs):
                            # Calculate the x offset for the second agent with a gap of 50 pixels
                            x_offset = x * 250 + 50

                            # Calculate the color intensity based on the probability (higher probability means whiter color)
                            color_intensity = int(prob * 255)
                            color = (color_intensity, color_intensity, color_intensity)

                            # Draw a rectangle for each action (the height of the rectangle is proportional to the probability)
                            cv2.rectangle(map_img, (x_offset + j*25, map_img.shape[0] - 25), (x_offset + (j+1)*25, map_img.shape[0] - 25 - int(prob*100)), color, -1)

                            # Write the name of the action (UP, DOWN, LEFT, RIGHT) at the bottom of the rectangle
                            cv2.putText(map_img, ["U", "D", "L", "R"][j], (x_offset + j*25 + 5, map_img.shape[0] - 5), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            

            # If rewards earned by the pacman agents are available, display them
            # if infos['rewards_earned'] is not None :
            #     # Add the rewards on the left of the image. Like a list of rewards for each pacman agent
            #     for i, reward in enumerate(infos['rewards_earned']):
            #         cv2.putText(map_img, f"J{i} reward : {reward}", (20, map_img.shape[0] - 20 - i*20), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return map_img




