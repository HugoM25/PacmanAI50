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
# batch_size = 32
# epochs = 4
# gae_lambda = 0.95


# #  train level 10 - 1
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
# learning_rate = 1e-5
# gamma = 0.95
# clip_epsilon = 0.2
# entropy_coefficient = 0.01
# value_coefficient = 0.2
# buffer_size = 4096
# batch_size = 64
# epochs = 4
# gae_lambda = 0.95


# Exp lvl 0 M4 128 max
# learning_rate = 3e-4
# gamma = 0.99
# clip_epsilon = 0.2
# entropy_coefficient = 0.01
# value_coefficient = 0.5
# buffer_size = 2048
# batch_size = 32
# epochs = 2
# gae_lambda = 0.95

# big level
# learning_rate = 3e-4
# gamma = 0.99
# clip_epsilon = 0.2
# entropy_coefficient = 0.05
# value_coefficient = 0.5
# buffer_size = 2048
# batch_size = 32
# epochs = 4
# gae_lambda = 0.95

#last used
# learning_rate = 3e-4
# gamma = 0.99
# clip_epsilon = 0.2
# entropy_coefficient = 0.1
# value_coefficient = 0.5
# buffer_size = 4096
# batch_size = 64
# epochs = 4
# gae_lambda = 0.95

# learning_rate = 3e-4
# gamma = 0.99
# clip_epsilon = 0.2
# entropy_coefficient = 0.05  # Increase entropy coefficient
# value_coefficient = 0.5
# buffer_size = 2048
# batch_size = 64
# epochs = 2
# gae_lambda = 0.95


#good idea maybe for big level
# learning_rate = 1e-5
# gamma = 0.99
# clip_epsilon = 0.2
# entropy_coefficient = 0.01
# value_coefficient = 0.5
# buffer_size = 2048
# batch_size = 64
# epochs = 4
# gae_lambda = 0.95


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
    #level_list_to_use = ['pacman_game/res/levels/level_0.csv', 'pacman_game/res/levels/level_00.csv']
    #level_list_to_use = ['pacman_game/res/levels/level_3-0.csv', 'pacman_game/res/levels/level_3-1.csv','pacman_game/res/levels/level_3-2.csv','pacman_game/res/levels/level_3-3.csv']
    #level_list_to_use = ['pacman_game/res/levels/level_1_2.csv', 'pacman_game/res/levels/level_1_2_1.csv','pacman_game/res/levels/level_1_2_2.csv']
    #level_list_to_use = ['pacman_game/res/levels/level_10_2.csv']
    #◘level_list_to_use = ['pacman_game/res/levels/level_2.csv', 'pacman_game/res/levels/level_2_1.csv', 'pacman_game/res/levels/level_2_2.csv']
    #level_list_to_use = ["pacman_game/res/levels/level_10.csv", "pacman_game/res/levels/level_11.csv", "pacman_game/res/levels/level_12.csv", "pacman_game/res/levels/level_13.csv", "pacman_game/res/levels/level_14.csv", "pacman_game/res/levels/level_15.csv"]
    #level_4 = ["pacman_game/res/levels/level_4-0.csv"]

    level_list_to_use = ['pacman_game/res/levels/final_level.csv']

    #level_list_to_use = ['pacman_game/res/levels/level_intro.csv']

    #level_list_to_use = ['pacman_game/res/levels/level9-0.csv',"pacman_game/res/levels/level9-1.csv", "pacman_game/res/levels/level9-2.csv", "pacman_game/res/levels/level9-3.csv", "pacman_game/res/levels/level9-4.csv"]
    
    environment = PacmanEnv(levels_paths=level_list_to_use, freq_change_level=1)

    # Initialize the trainer
    model = PacmanModelPPO(environment.observation_space.shape, 4)
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
                         n_steps=buffer_size,
                         use_action_masks=True,
                         mask_penalty=1.0,
                         show_gameplay_freq=100,
                         save_video_freq=50,
                         save_model_freq=50_000,
                         max_steps_env=1024,
                         max_grad_norm=0.1)


    # trainer.load_model("human_trained_model.pth")

    # Current train order
    # LEVEL 0
    # LEVEL 3 (no ghosts)
    # LEVEL 10 (no ghosts)
    
    # BEST MODEL FOR LEVEL 0
    # trainer.load_model("models/EXP_Lvl0_M4/model2001309.pth")

    # GOOD MODEL FOR LEVEL 3
    #trainer.load_model("models/EXP_level31_M4/model622184.pth")


    #trainer.load_model("models/exp_big_level_m4/model5935824.pth")
    #trainer.load_model("models/EXP_level10_m4/model2376000.pth")
    #trainer.load_model("models/M4_VID_LVL0/model_best.pth")
    #trainer.load_model("models/EXP_level10_6/model585728.pth")

    # trainer.load_model("models/lvl0_m5/model161645.pth")

    #trainer.load_model("models/VID_TRAIN_1/model1750007.pth")
    #trainer.load_model("models/VID_TRAIN_3/model1965600.pth")
    #trainer.load_model("models/VID_TRAIN_4/model1058400.pth")
    #trainer.load_model("models/VID_TRAIN_7/model457349.pth")
    
    
    #trainer.load_model("models/M4_VID_ok/model_best.pth")
    #trainer.load_model("models/M4_vid_BIG_lvl_night/model_best.pth")

    # ---------

    #trainer.load_model("models/M4_VID_BIG_LEVEL_GOOD_START/model_best.pth")
    trainer.load_model("models/main_lvl_M4_vid/model_best.pth")
    #trainer.load_model("models/level9_M4_VID/model_best.pth")

    # trainer.load_model("models/lvl_gh_VID_M4/model_best.pth")



    # Perform training
    trainer.train(20_000_000)

