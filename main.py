from pacman_game import PacmanEnv
from trainers import DQNTrainer, PPOTrainer, NewPPOTrainer
from models import *


# Hyperparameters
learning_rate = 3e-4
gamma = 0.99
clip_epsilon = 0.2
entropy_coefficient = 0.001
value_coefficient = 0.7
n_steps = 128
batch_size = 32
epochs = 4

if __name__ == "__main__":

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the environment
    environment = PacmanEnv("pacman_game/res/level11/")

    # Initialize the trainer
    model = ConvActorCritic(environment.observation_space.shape, 4)
    # model.apply(initialize_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    trainer = PPOTrainer(environment, model=model,
                         optimizer=optimizer, 
                         device=device,
                         clip_epsilon=clip_epsilon, 
                         gamma=gamma,
                         entropy_coef=entropy_coefficient, 
                         value_coef=value_coefficient, 
                         epochs=epochs, 
                         batch_size=batch_size, 
                         n_steps=n_steps)
    
    #trainer = DQNTrainer(environment, model=NNDQN(environment))

    # Perform training
    trainer.train(50000)

