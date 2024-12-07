from pacman_game import PacmanEnv
from trainers import DQNTrainer, PPOTrainer
from models import *


# Hyperparameters
learning_rate = 1e-5
gamma = 0.99
clip_epsilon = 0.2
entropy_coefficient = 0.2
value_coefficient = 0.2
n_steps = 128
batch_size = 32
epochs = 1



if __name__ == "__main__":

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the environment
    level_list_to_use = ["pacman_game/res/level11/", "pacman_game/res/level12/", "pacman_game/res/level13/", "pacman_game/res/level14/"]

    environment = PacmanEnv(levels_paths=level_list_to_use)

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

    # trainer.load_model("human_trained_model.pth")


    # Perform training
    trainer.train(50000)

