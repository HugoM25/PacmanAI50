import numpy as np
import random
from pacman_game.map import *
from pacman_game.algorithms import NavigationAlgo
from pacman_game.constants import *
from pacman_game.pacman import Pacman

class Ghost:
    def __init__(self, position: tuple[int, int]):
        self.start_position = position
        self.position = self.start_position
        self.is_free = True
        self.pacman_target = None
        self.detection_range = 10000  # Portée maximale de détection de Pacman


    def choose_direction(self, pacmans: list[Pacman], navigation_algo:NavigationAlgo, ghosts) -> int:
        '''
        Choose the direction of the ghost (0: up, 1: down, 2: left, 3: right)
        '''
        pass

    def reset(self):
        self.position = self.start_position
        self.is_free = True

    def find_pacman_target(self, pacmans: list[Pacman], navigation_algo:NavigationAlgo) -> Pacman:
        '''
        Find the pacman the ghost will target
        '''
        closest_pacman = None
        closest_distance = float('inf')

        for pacman in pacmans:
            distance = navigation_algo.heuristic(self.position, pacman.position)
            if distance < closest_distance:
                closest_distance = distance
                closest_pacman = pacman

        return closest_pacman, closest_distance

    def get_action_path(self, target_position, navigation_algo:NavigationAlgo):
        '''
        Get the action to take to reach the target position and the full path
        '''
        direction = -1
        path = navigation_algo.find_shortest_path_a_star(self.position, target_position)
        if path and len(path) > 1:
            next_step = path[0]
            if next_step[0] < self.position[0]:
                direction = UP
            elif next_step[0] > self.position[0]:
                direction = DOWN
            elif next_step[1] < self.position[1]:
                direction = LEFT
            elif next_step[1] > self.position[1]:
                direction = RIGHT

        return direction, path


class Blinky(Ghost):
    def __init__(self, position: tuple[int, int]):
        super().__init__(position=position)
        self.ghost_type = GHOST_BLINKY

    def choose_direction(self, pacmans: list[Pacman], navigation_algo:NavigationAlgo, ghosts) -> int:
        """
        Blinky est le fantôme rouge. Il poursuit le Pacman le plus proche en utilisant A* s'il est à portée, sinon il explore.
        """

        path = None

        # Find the closest Pacman on the map
        closest_pacman, closest_distance = self.find_pacman_target(pacmans, navigation_algo)

        direction = -1

        # If the closest Pacman is in the detection range, use A* to find the shortest path
        if closest_distance <= self.detection_range:
            direction, path = self.get_action_path(closest_pacman.position, navigation_algo)

        if direction == -1 :
            direction = random.randint(0, 3)

        # If the closest Pacman is not in the detection range, move randomly
        return direction, path

class Pinky(Ghost):
    def __init__(self, position: tuple[int, int]):
        super().__init__(position=position)
        self.ghost_type = GHOST_PINKY
        self.target = None

    def choose_direction(self, pacmans: list[Pacman], navigation_algo:NavigationAlgo, ghosts) -> int:
        '''
        Pinky is the pink ghost. She tries to ambush Pacman.
        Pour le moment, Pinky se déplace aléatoirement, mais accepte les positions des Pacmans.
        '''
        path = None

        # Find the closest Pacman on the map
        closest_pacman, closest_distance = self.find_pacman_target(pacmans, navigation_algo)

        # Tries to ambush Pacman by targeting the position n tiles in front of him
        # Only if pacman is more than 2 tiles away
        if closest_distance > 2:
            tiles_ahead = 2
            direction_pacman = ACTION_MAP[closest_pacman.last_action]
            target = (closest_pacman.position[0] + tiles_ahead * direction_pacman[0], closest_pacman.position[1] + tiles_ahead * direction_pacman[1])

            if target[0] < MAP_HEIGHT and target[1] < MAP_WIDTH:
                self.target = target

            if self.target ==  None:
                return random.randint(0, 3), path
        else :
            self.target = closest_pacman.position

        direction = -1

        if self.detection_range >= closest_distance:
            direction, path = self.get_action_path(self.target, navigation_algo)

        if direction == -1 :
            direction = random.randint(0, 3)

        # If the closest Pacman is not in the detection range, move randomly
        return direction, path


class Inky(Ghost):
    def __init__(self, position: tuple[int, int]):
        super().__init__(position=position)
        self.ghost_type = GHOST_INKY

    def choose_direction(self, pacmans: list[Pacman], navigation_algo:NavigationAlgo, ghosts) -> int:
        '''
        Inky is the blue ghost. He tries to ambush Pacman.
        Pour le moment, Inky se déplace aléatoirement, mais accepte les positions des Pacmans.
        '''
        path = None

        # Find the closest Pacman on the map
        closest_pacman, closest_distance = self.find_pacman_target(pacmans, navigation_algo)

        direction = -1

        # If the closest Pacman is in the detection range, use A* to find the shortest path
        if closest_distance <= self.detection_range:
            direction, path = self.get_action_path(closest_pacman.position, navigation_algo)

        if direction == -1 :
            direction = random.randint(0, 3)

        # If the closest Pacman is not in the detection range, move randomly
        return direction, path


class Clyde(Ghost):
    def __init__(self, position: tuple[int, int]):
        super().__init__(position=position)
        self.ghost_type = GHOST_CLYDE

    def choose_direction(self, pacmans: list[Pacman], navigation_algo:NavigationAlgo, ghosts) -> int:
        '''
        Clyde is the orange ghost. He does random moves.
        Pour le moment, Clyde se déplace aléatoirement, mais accepte les positions des Pacmans.
        '''
        path = None

        return random.randint(0, 3), path
