from pacman_game import PacEnv
from dqn_trainer import DQNTrainer, PolicyNetwork2

if __name__ == "__main__":
    environment = PacEnv("pacman_game/res/level0/", flatten_observation=True)
    trainer = DQNTrainer(environment, model=PolicyNetwork2(environment))
    trainer.train()