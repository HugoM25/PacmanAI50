import numpy as np
import random
from pacman_game.astar import choose_pacman_to_chase, astar

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

class Ghost:
    def __init__(self, position: tuple[int, int]):
        self.start_position = position
        self.position = self.start_position
        self.is_free = True

    def choose_direction(self, map: np.array, pacman_positions: list[tuple[int, int]]) -> int:
        '''
        Choose the direction of the ghost (0: up, 1: down, 2: left, 3: right)
        '''
        pass

    def reset(self):
        self.position = self.start_position
        self.is_free = True


class Blinky(Ghost):
    def __init__(self, position: tuple[int, int]):
        super().__init__(position=position)
        self.ghost_type = GHOST_BLINKY
        self.detection_range = 10  # Portée maximale de détection de Pacman

    def choose_direction(self, map: np.array, pacman_positions: list[tuple[int, int]]) -> int:
        """
        Blinky est le fantôme rouge. Il poursuit le Pacman le plus proche en utilisant A* s'il est à portée, sinon il explore.
        """
        # Choisir le Pacman le plus proche à poursuivre
        target_pacman = choose_pacman_to_chase(pacman_positions, tuple(self.position), map, self.detection_range)

        if target_pacman is not None:
            # Utilise A* pour poursuivre le Pacman si celui-ci est à portée
            path = astar(map, tuple(self.position), target_pacman)
            if len(path) > 1:
                next_position = path[1]  # Le premier élément est la position actuelle
                # Calculer la direction à prendre pour atteindre la prochaine position
                delta = (next_position[0] - self.position[0], next_position[1] - self.position[1])
                if delta == (-1, 0):
                    return 0  # Haut
                elif delta == (1, 0):
                    return 1  # Bas
                elif delta == (0, -1):
                    return 2  # Gauche
                elif delta == (0, 1):
                    return 3  # Droite

        # Si A* ne trouve pas de chemin, choisir une direction heuristique
        if target_pacman is not None:
            delta = (target_pacman[0] - self.position[0], target_pacman[1] - self.position[1])
            if abs(delta[0]) > abs(delta[1]):
                if delta[0] > 0:
                    return 1  # Bas
                else:
                    return 0  # Haut
            else:
                if delta[1] > 0:
                    return 3  # Droite
                else:
                    return 2  # Gauche

        # Déplacement aléatoire s'il ne détecte pas de Pacman à portée
        return random.randint(0, 3)


class Pinky(Ghost):
    def __init__(self, position: tuple[int, int]):
        super().__init__(position=position)
        self.ghost_type = GHOST_PINKY

    def choose_direction(self, map: np.array, pacman_positions: list[tuple[int, int]]) -> int:
        '''
        Pinky is the pink ghost. She tries to ambush Pacman.
        Pour le moment, Pinky se déplace aléatoirement, mais accepte les positions des Pacmans.
        '''
        return random.randint(0, 3)


class Inky(Ghost):
    def __init__(self, position: tuple[int, int]):
        super().__init__(position=position)
        self.ghost_type = GHOST_INKY

    def choose_direction(self, map: np.array, pacman_positions: list[tuple[int, int]]) -> int:
        '''
        Inky is the blue ghost. He tries to ambush Pacman.
        Pour le moment, Inky se déplace aléatoirement, mais accepte les positions des Pacmans.
        '''
        return random.randint(0, 3)


class Clyde(Ghost):
    def __init__(self, position: tuple[int, int]):
        super().__init__(position=position)
        self.ghost_type = GHOST_CLYDE

    def choose_direction(self, map: np.array, pacman_positions: list[tuple[int, int]]) -> int:
        '''
        Clyde is the orange ghost. He does random moves.
        Pour le moment, Clyde se déplace aléatoirement, mais accepte les positions des Pacmans.
        '''
        return random.randint(0, 3)
