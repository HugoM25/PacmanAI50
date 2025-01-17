import numpy as np

class Pacman() :
    def __init__(self, position: tuple[int, int]):
        super().__init__()

        # Animation variables
        self.death_anim = [4,5,6,7,8,9,10,11]
        self.idle_anim = [1,2]

        self.animation_frame = 0
        self.current_animation = self.idle_anim
        self.animation_speed = 0.025

        # Position variables
        self.position = position
        self.start_pos = position

        self.pacgum_eaten = 0

        self.score = 0

        self.alive = True

        self.superpower_step_left = 0

        self.last_action = -1

        self.has_key = False

        self.exploration_map = np.zeros((31, 28))

    def reset(self) :
        self.position = self.start_pos
        self.score = 0
        self.current_animation = self.idle_anim
        self.alive = True
        self.pacgum_eaten = 0
        self.superpower_step_left = 0
        self.last_action = -1
        self.has_key = False

        self.exploration_map = np.zeros((31, 28))




