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

# REWARDS = {
#     "RW_GUM": 200,
#     "RW_SUPER_GUM": 100,
#     "RW_EMPTY": -50,
#     "RW_NO_MOVE": -110,
#     "RW_DYING_TO_GHOST": -10110,
#     "RW_EATING_GHOST": 500,
#     "RW_WINNING": 10000,
#     "RW_TURNING_BACK": -100,
#     "RW_KEY": 120,
#     "RW_LIVING": -5
# }

# REWARDS = {
#     "RW_GUM": 154,
#     "RW_SUPER_GUM": 200,
#     "RW_EMPTY": -7,
#     "RW_NO_MOVE": -110,
#     "RW_DYING_TO_GHOST": -10110,
#     "RW_EATING_GHOST": 500,
#     "RW_WINNING": 10000,
#     "RW_TURNING_BACK": -100,
#     "RW_KEY": 120,
#     "RW_LIVING": -5
# }

REWARDS = {
    "RW_GUM": 40,  # Increase reward for eating gum
    "RW_SUPER_GUM": 50,  # Increase reward for eating super gum
    "RW_EMPTY": 0,  # Small penalty for empty moves
    "RW_NO_MOVE": -505,  # Penalty for no move
    "RW_DYING_TO_GHOST": -1000,  # Penalty for dying to ghost
    "RW_EATING_GHOST": 200,  # Reward for eating ghost
    "RW_WINNING": 10000,  # Reward for winning
    "RW_TURNING_BACK": -5,  # Small penalty for turning back
    "RW_KEY": 20,  # Reward for picking up key
    "RW_LIVING": 1,  # Small penalty for each step to encourage faster completion
    "RW_FRUIT": 45, # Reward for eating fruit
    "EXPLORE_REWARD": 5,  # Reward for exploring new area
    "ALREADY_EXPLORED": -0.5  # Penalty for already explored area
}

MAX_STEPS = 2048

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