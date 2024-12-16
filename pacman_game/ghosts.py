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
        self.prob_random_move = 0.05

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

            if pacman.alive == False:
                continue

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
        direction = -1,
        path = None

        # Find the closest Pacman on the map
        closest_pacman, closest_distance = self.find_pacman_target(pacmans, navigation_algo)

        # If no Pacman is found, move randomly (no target)
        if closest_pacman is None:
            return random.randint(0, 3), path

        self.pacman_target = closest_pacman

        # If the closest Pacman is in the detection range, use A* to find the shortest path
        if closest_distance <= self.detection_range:
            direction, path = self.get_action_path(closest_pacman.position, navigation_algo)

            if direction == -1 :
                # Move randomly
                direction = random.randint(0, 3)
            elif closest_pacman.superpower_step_left > 0:
                if random.random() < self.prob_random_move :
                    # Move randomly
                    direction = random.randint(0, 3)
                else :
                    # If the Pacman is in super, the ghost will run away
                    direction = OPPOSITE_ACTION[direction]

        return direction, path

class Pinky(Ghost):
    def __init__(self, position: tuple[int, int]):
        super().__init__(position=position)
        self.ghost_type = GHOST_PINKY
        self.target_pos = None

    def choose_direction(self, pacmans: list[Pacman], navigation_algo:NavigationAlgo, ghosts) -> int:
        '''
        Pinky is the pink ghost. She tries to ambush Pacman.
        Pour le moment, Pinky se déplace aléatoirement, mais accepte les positions des Pacmans.
        '''
        direction = -1,
        path = None
        tiles_ahead = 3
        distance_chase = 2

        # Find the closest Pacman on the map
        closest_pacman, closest_distance = self.find_pacman_target(pacmans, navigation_algo)

        # If no Pacman is found, move randomly (no target)
        if closest_pacman is None:
            return random.randint(0, 3), path

        self.pacman_target = closest_pacman

        # If pacman is too far away, tries to ambush it
        if closest_distance > distance_chase:

            # Get the direction of the closest pacman to go in front of it
            direction_pacman = ACTION_MAP[closest_pacman.last_action]
            target = (closest_pacman.position[0] + tiles_ahead * direction_pacman[0], closest_pacman.position[1] + tiles_ahead * direction_pacman[1])

            # If the target position is within the map boundaries and not a wall, set it as the target
            self.target_pos = target
            # if no target is set, move randomly
            if self.target_pos is None:
                return random.randint(0, 3), path
        else :
            # When the pacman is close enough, Pinky will target the pacman directly
            self.target_pos = closest_pacman.position

        # If the closest Pacman is in the detection range, use A* to find the shortest path
        if closest_distance <= self.detection_range:
            direction, path = self.get_action_path(self.target_pos, navigation_algo)

            if direction == -1 :
                # Move randomly
                direction = random.randint(0, 3)
            elif closest_pacman.superpower_step_left > 0:
                if random.random() < self.prob_random_move :
                    # Move randomly
                    direction = random.randint(0, 3)
                else :
                    # If the Pacman is in super, the ghost will run away
                    direction = OPPOSITE_ACTION[direction]
        # If the closest Pacman is not in the detection range, move randomly
        return direction, path


class Inky(Ghost):
    def __init__(self, position: tuple[int, int]):
        super().__init__(position=position)
        self.ghost_type = GHOST_INKY

    def choose_direction(self, pacmans: list[Pacman], navigation_algo:NavigationAlgo, ghosts) -> int:
        '''
        Inky is the blue ghost. He tries to ambush Pacman by positioning himselft at the opposite of Blinky.
        Pour le moment, Inky se déplace aléatoirement, mais accepte les positions des Pacmans.
        '''
        path = None
        direction = -1
        take_blinky_into_account = False

        # Find the closest Pacman on the map
        closest_pacman, closest_distance = self.find_pacman_target(pacmans, navigation_algo)

        # If no Pacman is found, move randomly (no target)
        if closest_pacman is None:
            return random.randint(0, 3), path

        self.pacman_target = closest_pacman

        # Find the position of Blinky and check if blinky is tracking a pacman
        blinky = ghosts[0]
        if blinky.pacman_target is not None:
            if blinky.pacman_target == closest_pacman:
                take_blinky_into_account = True

        # If Blinky is tracking a pacman, Inky will try to ambush the pacman by positioning himself at the opposite of Blinky
        if take_blinky_into_account:
            blinky_position = blinky.position

            # Calculate the position of Inky (should be at the opposite of Blinky relative to the closest Pacman)
            inky_position = (closest_pacman.position[0] - (blinky_position[0] - closest_pacman.position[0]), closest_pacman.position[1] - (blinky_position[1] - closest_pacman.position[1]))


            # If the position of Inky is within the map boundaries and not a wall, set it as the target
            if inky_position[0] < MAP_HEIGHT and inky_position[1] < MAP_WIDTH and navigation_algo.current_map[inky_position[0]][inky_position[1]] != WALL:
                target_position = inky_position
            else:
                target_position = closest_pacman.position

        # Move towards target
        direction, path = self.get_action_path(target_position, navigation_algo)

        if direction == -1 :
            direction = random.randint(0, 3)

        # Check if the Pacman is currently in super mode
        if closest_pacman.superpower_step_left > 0:
            # If the Pacman is in super, the ghost will run away
            direction = OPPOSITE_ACTION[direction]

        # If the closest Pacman is not in the detection range, move randomly
        return direction, path


class Clyde(Ghost):
    def __init__(self, position: tuple[int, int]):
        super().__init__(position=position)
        self.ghost_type = GHOST_CLYDE
        self.target_position = None

    def find_random_target(self, navigation_algo:NavigationAlgo):
        '''
        Find a random target on the map
        '''
        target = (random.randint(0, MAP_HEIGHT - 1), random.randint(0, MAP_WIDTH - 1))
        while navigation_algo.current_map[target[0]][target[1]] == WALL:
            target = (random.randint(0, MAP_HEIGHT - 1), random.randint(0, MAP_WIDTH - 1))

        return target

    def choose_direction(self, pacmans: list[Pacman], navigation_algo:NavigationAlgo, ghosts) -> int:
        '''
        Clyde is the orange ghost. He does random moves.
        Pour le moment, Clyde se déplace aléatoirement, mais accepte les positions des Pacmans.
        '''
        path = None
        direction = -1

        # Find a random target on the map if no target is set
        if self.target_position is None:
            self.target_position = self.find_random_target(navigation_algo)
        else :
            # If the target is reached, find a new random target
            if self.position[0] == self.target_position[0] and self.position[1] == self.target_position[1]:
                self.target_position = self.find_random_target(navigation_algo)

            # If the target is not reached, move towards it
            direction, path = self.get_action_path(self.target_position, navigation_algo)

        if direction == -1:
            direction = random.randint(0, 3)

        return direction, path
