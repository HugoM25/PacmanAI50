import numpy as np
import random

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

class Ghost :
    def __init__(self,  position: tuple[int, int]):
        self.start_position = position
        self.position = self.start_position
        self.is_free = True

    def choose_direction(self, map: np.array) -> int:
        '''
        Choose the direction of the ghost (0: up, 1: down, 2: left, 3: right)
        '''
        pass

    def reset(self):
        self.position = self.start_position
        self.is_free = True


class Blinky(Ghost):
    def __init__(self,  position: tuple[int, int]):
        super().__init__(position=position)
        self.ghost_type =GHOST_BLINKY

    def choose_direction(self, map: np.array) -> int:
        '''
        Blinky is the red ghost. He chases Pacman.
        '''


        return random.randint(0, 3)



class Pinky(Ghost):
    def __init__(self,  position: tuple[int, int]):
        super().__init__(position=position)
        self.ghost_type =GHOST_PINKY

    def choose_direction(self, map: np.array) -> int:
        '''
        Pinky is the pink ghost. She tries to ambush Pacman.
        '''
        return random.randint(0, 3)


class Inky(Ghost):
    def __init__(self,  position: tuple[int, int]):
        super().__init__(position=position)
        self.ghost_type =GHOST_INKY


    def choose_direction(self, map: np.array) -> int:
        '''
        Inky is the blue ghost. He tries to ambush Pacman.
        '''
        return random.randint(0, 3)


class Clyde(Ghost):
    def __init__(self,  position: tuple[int, int]):
        super().__init__(position=position)
        self.ghost_type =GHOST_CLYDE

    def choose_direction(self, map: np.array) -> int:
        '''
        Clyde is the orange ghost. He does random moves.
        '''
        return random.randint(0, 3)