from pacman_game import PacmanEnv
from trainers import DQNTrainer, PPOTrainer, NewPPOTrainer
from models import *

if __name__ == "__main__":
    # Initialize the environment
    environment = PacmanEnv("pacman_game/res/level0/")

    # Initialize the trainer
    trainer = NewPPOTrainer(environment,
                         actor_model=ConvPPOActor(environment),
                         critic_model=ConvPPOCritic(environment))

    #trainer = DQNTrainer(environment, model=NNDQN(environment))

    # Perform training
    trainer.train(500000)

