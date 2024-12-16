import numpy as np

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTION_MAP = {
    UP: np.array([-1, 0]),  # Up
    DOWN: np.array([1, 0]),   # Down
    LEFT: np.array([0, -1]),  # Left
    RIGHT: np.array([0, 1]),   # Right
}

OPPOSITE_ACTION = {
    UP: DOWN,
    DOWN: UP,
    LEFT: RIGHT,
    RIGHT: LEFT
}

REWARDS = {
    "RW_GUM": 10,
    "RW_SUPER_GUM": 10,
    "RW_EMPTY": -0.0,
    "RW_NO_MOVE": -10,
    "RW_DYING_TO_GHOST": -10,
    "RW_EATING_GHOST": 10,
    "RW_WINNING": 10000,
    "RW_TURNING_BACK": -0.6,
    "RW_KEY": 7,
    "RW_LIVING": -0.5
}

MAX_STEPS = 1500

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
KEY = 12
DOOR = 13


MAP_WIDTH = 28
MAP_HEIGHT = 31