from pacman_game import PacmanEnv
from trainers import DQNTrainer, PPOTrainer
from models import *


learning_rate = 3e-4
gamma = 0.99
clip_epsilon = 0.2
entropy_coefficient = 0.1
value_coefficient = 0.5
buffer_size = 4096
batch_size = 64
epochs = 4
gae_lambda = 0.95

if __name__ == "__main__":

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the environment
    level_list_to_use = ['pacman_game/res/levels/level4_1.csv', 'pacman_game/res/levels/level4_2.csv', 'pacman_game/res/levels/level4_3.csv','pacman_game/res/levels/level4_4.csv']
    environment = PacmanEnv(levels_paths=level_list_to_use, freq_change_level=1)

    # Initialize the trainer
    model = PacmanModelPPO(environment.observation_space.shape, 4)

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
                         n_steps=buffer_size,
                         use_action_masks=True,
                         mask_penalty=1.0,
                         show_gameplay_freq=100,
                         save_video_freq=50,
                         save_model_freq=50_000,
                         max_steps_env=1024,
                         max_grad_norm=0.1)


    # Load a model if needed
    # trainer.load_model("path_to_your_model.pth")

    # Perform training
    trainer.train(5_000_000)

