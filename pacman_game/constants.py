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
    "RW_GUM": 0.6,
    "RW_SUPER_GUM": 0.8,
    "RW_EMPTY": 0.0,
    "RW_NO_MOVE": -0.5,
    "RW_DYING_TO_GHOST": -1,
    "RW_EATING_GHOST": 0.9,
    "RW_WINNING": 1,
    "RW_TURNING_BACK": -0.4,
    "RW_KEY": 0.7,
    "RW_LIVING": -0.05
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