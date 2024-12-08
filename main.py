from pacman_game import PacmanEnv
from trainers import DQNTrainer, PPOTrainer
from models import *

#  train level 0
# learning_rate = 3e-4
# gamma = 0.99
# clip_epsilon = 0.2
# entropy_coefficient = 0.01
# value_coefficient = 0.2
# buffer_size = 2048
# batch_size = 64
# epochs = 4
# gae_lambda = 0.95


#  train level 10 - 1
# learning_rate = 1e-4
# gamma = 0.95
# clip_epsilon = 0.2
# entropy_coefficient = 0.001
# value_coefficient = 0.2
# buffer_size = 2048
# batch_size = 64
# epochs = 4
# gae_lambda = 0.95

# train level 10 - 2
# learning_rate = 1e-5
# gamma = 0.95
# clip_epsilon = 0.2
# entropy_coefficient = 0.01
# value_coefficient = 0.2
# buffer_size = 4096
# batch_size = 64
# epochs = 4
# gae_lambda = 0.95

# Hyperparameters
learning_rate = 1e-5
gamma = 0.95
clip_epsilon = 0.2
entropy_coefficient = 0.01
value_coefficient = 0.2
buffer_size = 4096
batch_size = 64
epochs = 4
gae_lambda = 0.95

if __name__ == "__main__":

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the environment
    level_list_to_use = ["pacman_game/res/level11/", "pacman_game/res/level12/", "pacman_game/res/level13/", "pacman_game/res/level14/"]
    #level_list_to_use = ["pacman_game/res/level0/", "pacman_game/res/level00/"]
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
                         n_steps=buffer_size)

    # trainer.load_model("human_trained_model.pth")
    trainer.load_model("model_trained_5.pth")

    # Perform training
    trainer.train(10000000)

