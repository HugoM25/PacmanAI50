from pacman_game import PacmanEnv
from dqn_trainer import DQNTrainer
from ppo_trainer import PPOTrainer
from PacmanAI50.models import *

if __name__ == "__main__":
    # Initialize the environment
    environment = PacmanEnv("pacman_game/res/level0/")

    # Initialize the trainer
    trainer = PPOTrainer(environment,
                         actor_model=NNPPOActor(environment),
                         critic_model=NNPPOCritic(environment))

    #trainer = DQNTrainer(environment, model=NNDQN(environment))

    # Perform training
    trainer.train(50000)

