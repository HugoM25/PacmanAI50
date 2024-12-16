
import cv2
import numpy as np
from pynput import keyboard
import json

class HumanCollector:
    '''
    This class is used to collect manually inputted trajectories from a human player
    For every agent in the environment, the human player will be asked to input an action
    The human will be shown the current state of the environment and will be asked to input an action
    '''
    def __init__(self, env, data_file_path, iterations=100):
        '''
        Initialize the HumanCollector
        @param env: The environment
        @param data_file: The file to save the data
        '''
        self.env = env
        self.data_file_path = data_file_path
        self.iterations = iterations

        self.trajectories_collected = []

    def collect_trajectories(self):
        '''
        Collect trajectories from the human player
        '''

        # Reset the environment
        observations, _ = self.env.reset()
        done = False

        self.trajectory = []

        infos = {}

        ep_reward = 0

        for _ in range(self.iterations):

            actions_to_play = []
            for agent_index, observation in enumerate(observations):

                disp_info = {}
                if 'ghosts_paths' in infos and infos['ghosts_paths'] :
                    disp_info = {
                        'ghosts_paths': infos['ghosts_paths']
                    }
                # Render the environment as seen by the agent
                img = self.env.render(mode='rgb_array', infos=disp_info)
                print("Rendering image")
                cv2.imshow('Pacman', img)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break

                #Get the action from the human player
                action = self.get_human_action()
                actions_to_play.append(action)

            # Step the environment
            observations, rewards_earned, done, truncated, infos = self.env.step(actions_to_play)

            ep_reward += rewards_earned[0]
            print("Reward earned :" + str(rewards_earned[0]) + "Total reward: " + str(ep_reward))

            # Store the trajectory
            experience = {
                'observation': observations[0],
                'action': actions_to_play[0],
                'reward': rewards_earned[0],
                'done': done
            }

            self.trajectory.append(experience)

            if done or truncated:
                observations, _ = self.env.reset()

        self.save_trajectories()

    def get_human_action(self):
            '''
            Get an action from the human player using arrow keys
            @return action: The action inputted by the human player
            '''
            print("Press an arrow key for the agent's action: ")

            def on_press(key):
                try:
                    if key == keyboard.Key.up:
                        self.action = 0
                    elif key == keyboard.Key.down:
                        self.action = 1
                    elif key == keyboard.Key.left:
                        self.action = 2
                    elif key == keyboard.Key.right:
                        self.action = 3
                    return False  # Stop listener
                except AttributeError:
                    pass

            with keyboard.Listener(on_press=on_press) as listener:
                listener.join()

            return self.action


    def save_trajectories(self):
        '''
        Save the trajectories to a file
        '''
        # Convert all numpy arrays to lists
        for trajectory in self.trajectory:
            trajectory['observation'] = trajectory['observation'].tolist() if isinstance(trajectory['observation'], np.ndarray) else trajectory['observation']

        with open(self.data_file_path, 'w') as f:
            json.dump(self.trajectory, f)




