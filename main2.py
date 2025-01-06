from pacman_game import PacmanEnv
from trainers import DQNTrainer, PPOTrainer
from models import *


if __name__ == "__main__":

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the environment
    level_list_to_use = ['pacman_game/res/levels/level_0.csv', 'pacman_game/res/levels/level_00.csv']
    # level_list_to_use = ['pacman_game/res/levels/level_3-0.csv', 'pacman_game/res/levels/level_3-1.csv','pacman_game/res/levels/level_3-2.csv','pacman_game/res/levels/level_3-3.csv']
    # level_list_to_use = ['pacman_game/res/levels/level_1_2.csv']
    #level_list_to_use = ['pacman_game/res/levels/level_2.csv', 'pacman_game/res/levels/level_2_1.csv', 'pacman_game/res/levels/level_2_2.csv']
    #level_list_to_use = ["pacman_game/res/levels/level_10.csv", "pacman_game/res/levels/level_11.csv", "pacman_game/res/levels/level_12.csv", "pacman_game/res/levels/level_13.csv", "pacman_game/res/levels/level_14.csv"]
    environment = PacmanEnv(levels_paths=level_list_to_use, freq_change_level=1)

    # Initialize the trainer
    model = PacmanModel2DQN(environment.observation_space.shape, 4)
    # model.apply(initialize_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    trainer = DQNTrainer(environment, model,
                    gamma=0.99, batch_size=64, buffer_size=500000,  # Adjusted batch size and buffer size
                    epsilon_start=0.5, epsilon_end=0.01, epsilon_decay=50000,  # Adjusted epsilon decay
                    target_update_freq=1000, min_replay_size=5000, device=device,  # Adjusted min replay size
                    model_save_freq=10000, show_gameplay_freq=50, max_steps_env=128,use_double_dqn=False)

    # Perform training
    trainer.train(20_000_000)

