import gym
from gym import spaces
import numpy as np
import os
from PIL import Image
import pandas as pd
from gym.envs.registration import register
import random
register(
    id='StockEnv-v0',
    entry_point='stock_env:StockEnv',
)

class StockEnv(gym.Env):
    """A stock market simulation for training reinforcement learning agents."""
    metadata = {'render.modes': ['human']}

    def __init__(self, img_width=800, img_height=575, test=False, stock=None):
        super(StockEnv, self).__init__()
        self.img_width = img_width
        self.img_height = img_height
        self.action_space = spaces.Discrete(3)  # Three discrete actions available
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.img_height, self.img_width, 3),
                                            dtype=np.uint8)  # Image observation space
        self.test = test
        self.folder_path = "test/" if self.test else "train/"
        self.stock = stock



    def _read_image_by_index(self, index,):
        file_name = f"{index}.png"
        file_path = os.path.join(self.folder_image_path, file_name)
        if os.path.exists(file_path):
            img = Image.open(file_path).convert('L')
            img_array = np.array(img)
            # Add a new dimension at the front (1, 84, 84)
            return np.expand_dims(img_array, axis=0)
        else:
            return None

    def _read_reward_by_index(self, index):
        if index < len(self.rewards_df):
            return self.rewards_df.iloc[index]['Reward']
        else:
            return None


    def calculate_rewards(self, actions, negative_reward_multiplier=1.5):

        daily_ouputs = [self._read_reward_by_index(i) for i in range(self.num_images)]

        daily_ouputs = list(filter(lambda daily_ouput: daily_ouput is not None, daily_ouputs))

        # Translate actions ( 0 | 1 | 2 ) into ( -1 | 0 | 1 )
        action_map = [-1, 0, 1]
        actions = [action_map[action] for action in actions]

        # Rewards
        rewards = [daily_ouput * action * negative_reward_multiplier \
                   if daily_ouput * action < 0 else daily_ouput * action \
                   for daily_ouput, action in zip(daily_ouputs, actions)]

        return rewards

    def reset(self):

        subfolders = [f.name for f in os.scandir(self.folder_path) if f.is_dir()]
        assert subfolders, "No subfolders found in the main folder path."

        # Select a subfolder randomly
        selected_subfolder = random.choice(subfolders) if self.stock is None else  self.stock

        self.folder_image_path = os.path.join(self.folder_path, selected_subfolder)

        rewards_file_path = os.path.join(self.folder_image_path, "test.csv" if self.test else "train.csv")
        self.rewards_df = pd.read_csv(rewards_file_path)

        self.num_images = len([f for f in os.listdir(self.folder_image_path) if f.endswith(".png")])

        assert not self.num_images == 0, f"{selected_subfolder} has no images"
        self.states = [self._read_image_by_index(i) for i in range(self.num_images)]

        assert not any(state is None for state in self.states), "Batch size is bigger than the number of images."

        return self.states

