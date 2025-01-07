import pygame
import tkinter as tk
from tkinter import filedialog

import torch
from pacman_game import PacmanEnv
from models import *
import cv2

WINDOW_W, WINDOW_H = (400, 600)

BG_BUTTON_START = (21, 50, 67)
BG_BUTTON_LOAD = (21, 50, 67)
BG_WINDOWS = (40, 75, 99)
BG_DROPDOWN = (21, 50, 67)
BG_DROPDOWN_HOVER = (180, 184, 171)
BG_BUTTON_STOP = (21, 50, 67)
TEXT_COLOR = (244, 249, 233)

def main(args) :
    pass

class Button(pygame.sprite.Sprite):
    def __init__(self, text, pos=(0,0), size=(10,10),  background_color=BG_BUTTON_START, text_color=TEXT_COLOR, icon_path=None):
        super(Button, self).__init__()

        self.text = text
        self.background_color = background_color
        self.text_color = text_color
        self.pos = pos
        self.size = size
        self.icon = pygame.image.load(icon_path) if icon_path else None
        if self.icon:
            self.icon = pygame.transform.scale(self.icon, (self.size[1] - 20, self.size[1] - 20))  # Scale icon to fit within the button height

    def draw(self, win) :
        font = pygame.font.Font(None, 36)
        text = font.render(self.text, True, self.text_color)

        # Calculate the position for the text and icon
        if self.icon:
            icon_rect = self.icon.get_rect()
            text_rect = text.get_rect()
            total_width = icon_rect.width + 10 + text_rect.width
            start_x = self.pos[0] + (self.size[0] - total_width) // 2
            icon_rect.topleft = (start_x, self.pos[1] + (self.size[1] - icon_rect.height) // 2)
            text_rect.topleft = (icon_rect.right + 10, self.pos[1] + (self.size[1] - text_rect.height) // 2)
        else:
            text_rect = text.get_rect(center=(self.pos[0] + self.size[0] // 2, self.pos[1] + self.size[1] // 2))

        pygame.draw.rect(win, self.background_color, (self.pos[0], self.pos[1], self.size[0], self.size[1]))

        if self.icon:
            win.blit(self.icon, icon_rect)

        win.blit(text, text_rect)

    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            if x > self.pos[0] and x < self.pos[0] + self.size[0] and y > self.pos[1] and y < self.pos[1] + self.size[1]:
                return True
        return False

class Dropdown:
    def __init__(self, options, pos=(0,0), size=(10,10), background_color=BG_DROPDOWN, text_color=TEXT_COLOR):
        self.options = options
        self.pos = pos
        self.size = size
        self.background_color = background_color
        self.text_color = text_color
        self.selected_option = options[0]
        self.expanded = False

    def draw(self, win):
        font = pygame.font.Font(None, 36)
        if self.expanded:
            for i, option in enumerate(self.options):
                bg_color = BG_DROPDOWN_HOVER if i == self.options.index(self.selected_option) else self.background_color
                text = font.render(option, True, self.text_color)
                text_rect = text.get_rect(center=(self.pos[0] + self.size[0]//2, self.pos[1] + self.size[1]//2 + i * self.size[1]))
                pygame.draw.rect(win, bg_color, (self.pos[0], self.pos[1] + i * self.size[1], self.size[0], self.size[1]))
                pygame.draw.rect(win, TEXT_COLOR, (self.pos[0], self.pos[1] + i * self.size[1], self.size[0], self.size[1]), 2)
                win.blit(text, text_rect)
        else:
            text = font.render(self.selected_option, True, self.text_color)
            text_rect = text.get_rect(center=(self.pos[0] + self.size[0]//2, self.pos[1] + self.size[1]//2))
            pygame.draw.rect(win, self.background_color, (self.pos[0], self.pos[1], self.size[0], self.size[1]))
            pygame.draw.rect(win, TEXT_COLOR, (self.pos[0], self.pos[1], self.size[0], self.size[1]), 2)
            win.blit(text, text_rect)
            # Draw the arrow
            arrow_color = TEXT_COLOR
            arrow_points = [
                (self.pos[0] + self.size[0] - 20, self.pos[1] + self.size[1]//2 - 5),
                (self.pos[0] + self.size[0] - 10, self.pos[1] + self.size[1]//2 - 5),
                (self.pos[0] + self.size[0] - 15, self.pos[1] + self.size[1]//2 + 5)
            ]
            pygame.draw.polygon(win, arrow_color, arrow_points)

    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            if self.pos[0] <= x <= self.pos[0] + self.size[0] and self.pos[1] <= y <= self.pos[1] + self.size[1] * (len(self.options) if self.expanded else 1):
                return True
        return False

    def handle_event(self, event):
        if self.is_clicked(event):
            if self.expanded:
                index = (event.pos[1] - self.pos[1]) // self.size[1]
                self.selected_option = self.options[index]
            self.expanded = not self.expanded

class Slider:
    def __init__(self, min_val, max_val, pos, size, initial_val=0, show_value=False):
        self.min_val = min_val
        self.max_val = max_val
        self.pos = pos
        self.size = size
        self.value = initial_val
        self.handle_pos = self.pos[0] + (self.value - self.min_val) / (self.max_val - self.min_val) * self.size[0]
        self.dragging = False
        self.show_value = show_value

    def draw(self, win):
        pygame.draw.rect(win, (100, 100, 100), (*self.pos, *self.size))
        pygame.draw.rect(win, (200, 200, 200), (self.handle_pos - 5, self.pos[1], 10, self.size[1]))
        if self.show_value:
            font = pygame.font.SysFont(None, 24)
            value_surf = font.render(f'{self.value:.2f}', True, (255, 255, 255))
            win.blit(value_surf, (self.pos[0] + self.size[0] + 10, self.pos[1]))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.pos[0] <= event.pos[0] <= self.pos[0] + self.size[0] and self.pos[1] <= event.pos[1] <= self.pos[1] + self.size[1]:
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                self.handle_pos = max(self.pos[0], min(event.pos[0], self.pos[0] + self.size[0]))
                self.value = self.min_val + (self.handle_pos - self.pos[0]) / self.size[0] * (self.max_val - self.min_val)


class TextField:
    def __init__(self, pos, size, initial_text='', font_size=24):
        self.pos = pos
        self.size = size
        self.text = initial_text
        self.font_size = font_size
        self.font = pygame.font.SysFont(None, self.font_size)
        self.active = False
        self.color_inactive = pygame.Color('lightskyblue3')
        self.color_active = pygame.Color('dodgerblue2')
        self.color = self.color_inactive
        self.rect = pygame.Rect(self.pos, self.size)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
            else:
                self.active = False
            self.color = self.color_active if self.active else self.color_inactive
        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    print(self.text)
                    self.text = ''
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode

    def draw(self, win):
        txt_surface = self.font.render(self.text, True, self.color)
        win.blit(txt_surface, (self.rect.x+5, self.rect.y+5))
        pygame.draw.rect(win, self.color, self.rect, 2)

class PacmanAI:
    def __init__(self):
        self.is_game_running = False
        self.model_type = "PPO"
        self.model_path = None

        self.selected_level = "Level 1"

        self.levels_available = { "Level 1": ['pacman_game/res/levels/level1_0.csv'],
                                  "Level 2": ['pacman_game/res/levels/level2_0.csv'],
                                  "Level 3": ['pacman_game/res/levels/level3_0.csv'],
                                  "Level 4": ["pacman_game/res/levels/level4_0.csv"],
                                  "Level Final": ["pacman_game/res/levels/final_level.csv"]
                                }

        self.model_types = ["PPO", "DQN"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = PacmanEnv(levels_paths=self.levels_available[self.selected_level], freq_change_level=1)
        self.env.max_steps = 2000

        if self.model_type == "PPO":
            self.model = PacmanModelPPO(self.env.observation_space.shape, 4).to(self.device)
        elif self.model_type== "DQN":
            self.model = PacmanModelDQN(self.env, 4).to(self.device)

        self.run_pygame_GUI()

    def run_pygame_GUI(self):
        # Initialize the window
        pygame.init()
        screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Pacman AI")
        running = True

        # Define the buttons
        button_load = Button("Load a Model", pos=(0, 170), size=(WINDOW_W, 50), background_color=BG_BUTTON_LOAD)
        dropdown_level = Dropdown([level_name for level_name in self.levels_available], pos=(0, 110), size=(WINDOW_W//2-5, 50))
        dropdown_model_type = Dropdown(["PPO", "DQN"], pos=(WINDOW_W//2+5, 110), size=(WINDOW_W//2-5, 50))
        button_stop = Button("STOP", pos=(0, 230), size=(WINDOW_W//2-5, 50))
        button_pause = Button("PAUSE", pos=(WINDOW_W//2+5, 230), size=(WINDOW_W//2-5, 50))

        button_start = Button("Start", pos=(0, 230), size=(WINDOW_W, 50), background_color=BG_BUTTON_STOP)

        episode_steps = 0
        done = False
        self.last_proba_actions = [[] for _ in range(self.env.nb_agents)]

        infos = {}


        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif dropdown_model_type.is_clicked(event):
                    dropdown_model_type.handle_event(event)

                    if dropdown_model_type.selected_option == "PPO":
                        self.model_type = "PPO"
                        self.model = PacmanModelPPO(self.env.observation_space.shape, 4).to(self.device)
                    elif dropdown_model_type.selected_option == "DQN":
                        self.model_type = "DQN"
                        self.model = PacmanModelDQN(self.env, 4).to(self.device)

                elif dropdown_level.is_clicked(event):
                    dropdown_level.handle_event(event)
                    self.selected_level = dropdown_level.selected_option
                elif button_load.is_clicked(event):
                    file_path = self.load_file()
                    if file_path is not None and ".pth" in file_path:
                        button_load.text = file_path.split("/")[-1]
                        self.load_model(file_path)
                        print(f"Model loaded: {file_path}")
                elif self.is_game_running and button_stop.is_clicked(event):
                    self.is_game_running = False
                elif self.is_game_running and button_pause.is_clicked(event):
                    self.is_game_running = False
                elif button_start.is_clicked(event):
                    observations, _ = self.start_game()

            # Fill the screen with a color
            screen.fill(BG_WINDOWS)

            # Draw the buttons and dropdown
            button_load.draw(screen)

            # Draw the buttons to launch the game (if the game is not running already)
            if  self.is_game_running:
                button_stop.draw(screen)
                button_pause.draw(screen)
            else :
                button_start.draw(screen)

            dropdown_level.draw(screen)
            dropdown_model_type.draw(screen)


            # Run the game loop (if the game is running)
            if self.is_game_running:
                if not done :
                    episode_steps += 1

                    disp_infos = {'step': episode_steps,
                                    'probabilities_moves': self.last_proba_actions,
                                    }

                    if 'ghosts_paths' in infos and infos['ghosts_paths'] :
                        disp_infos['ghosts_paths']= infos['ghosts_paths']


                    img = self.env.render(mode='rgb_array', infos=disp_infos)
                    self.show_gameplay(img)

                    actions_to_play = []

                    # For each agent, get the action to play
                    for agent_index, observation in enumerate(observations):
                            map_obs, info_obs = observation

                            map_state_tensor = torch.tensor(map_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                            info_state_tensor = torch.tensor(info_obs, dtype=torch.float32).unsqueeze(0).to(self.device)


                            if self.model_type == "PPO":
                                logits, value = self.model(map_state_tensor, info_state_tensor)

                                # Apply a mask to the logits to prevent invalid moves
                                action_mask = self.get_action_mask(map_state_tensor, info_state_tensor)

                                # Apply the mask to the logits
                                logits[~action_mask] = -float('inf')

                                dist = torch.distributions.Categorical(logits=logits)

                                self.last_proba_actions[agent_index] = dist.probs.detach().cpu().numpy()


                                action = dist.sample()
                                actions_to_play.append(action.item())
                            elif self.model_type == "DQN":
                                action = self.model(map_state_tensor, info_state_tensor)
                                # Get the action with the highest Q-value
                                actions_to_play.append(action.argmax().item())

                    # Step the environment
                    observations, rewards_earned, done, truncated, infos = self.env.step(actions_to_play)

                    if done or truncated:
                        episode_steps = 0
                        observations, _ = self.env.reset()
                        done = False

            # Update the display
            pygame.display.flip()

        pygame.quit()

    def show_gameplay(self, img):
        """
        Show the gameplay
        @param img: image to display
        """
        cv2.imshow('Pacman', img)
        # Simplified key check
        return cv2.waitKey(1) != ord('q')

    def start_game(self):

        if self.model_path is None :
            print("No model was loaded. Defaulting to random initiated model. Please load a model if you want to test one.")

        self.env = PacmanEnv(levels_paths=self.levels_available[self.selected_level], freq_change_level=1)
        self.last_proba_actions = [[] for _ in range(self.env.nb_agents)]
        self.env.max_steps = 2000
        self.is_game_running = True


        return self.env.reset()


    def load_file(self):
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        file_path = filedialog.askopenfilename()
        if file_path:
            return file_path
        return None

    def load_model(self, model_path):
            if model_path is not None:
                self.model_path = model_path
                try:
                    # Load the model onto the CPU regardless of its original device
                    pretrained_dict = torch.load(model_path, map_location=torch.device('cpu'))
                    self.model.load_state_dict(pretrained_dict)
                    self.model.to(self.device)  # Ensure the model is on the correct device
                    print(f"Model loaded: {model_path}")
                except Exception as e:
                    print(f"Error loading model: {e}")
            else:
                print("No model path provided. Please provide a model path to load a model.")

    def get_action_mask(self, map_state, info_state):
        '''
        Returns a mask for valid actions (True) and invalid actions (False)
        @param map_state: [batch_size, height, width] tensor containing the map
        @param info_state: [batch_size, ...] tensor containing agent info including position
        '''
        batch_size = map_state.shape[0]
        height, width = map_state.shape[1:3]
        masks = torch.ones((batch_size, 4), dtype=torch.bool, device=self.device)

        # Get current positions
        pos = info_state[:, 0:2].long()  # Get x,y positions
        batch_idx = torch.arange(batch_size, device=self.device)

        # Calculate wrapped positions
        up_pos = torch.remainder(pos[:, 0] - 1, height)
        down_pos = torch.remainder(pos[:, 0] + 1, height)
        left_pos = torch.remainder(pos[:, 1] - 1, width)
        right_pos = torch.remainder(pos[:, 1] + 1, width)

        # Check walls (type 9 or 10) at target positions
        # True = valid move, False = invalid move
        masks[:, 0] = ~(  # Up
            (map_state[batch_idx, up_pos, pos[:, 1]] == 9) |
            (map_state[batch_idx, up_pos, pos[:, 1]] == 10)
        )

        masks[:, 1] = ~(  # Down
            (map_state[batch_idx, down_pos, pos[:, 1]] == 9) |
            (map_state[batch_idx, down_pos, pos[:, 1]] == 10)
        )

        masks[:, 2] = ~(  # Left
            (map_state[batch_idx, pos[:, 0], left_pos] == 9) |
            (map_state[batch_idx, pos[:, 0], left_pos] == 10)
        )

        masks[:, 3] = ~(  # Right
            (map_state[batch_idx, pos[:, 0], right_pos] == 9) |
            (map_state[batch_idx, pos[:, 0], right_pos] == 10)
        )

        return masks

if __name__ == "__main__" :
    PacmanAI()