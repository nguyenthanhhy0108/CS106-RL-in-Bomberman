import os
import pickle
import random
import torch

import numpy as np
from settings import BOMB_TIMER


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        print("Loading")
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    return np.random.choice(ACTIONS, p=self.model)


def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to the input of your model, i.e. a feature vector.

    :param game_state: A dictionary describing the current game board.
    :return: np.array
    """
    if game_state is None:
        return None

    field = game_state['field']
    agent = game_state['self']
    others = game_state['others']
    bombs = game_state['bombs']
    coins = game_state['coins']

    nrows, ncols = field.shape
    
    field_channel = field
    agent_channel = np.zeros(field.shape)
    others_channel = np.zeros(field.shape)
    bombs_channel = np.zeros(field.shape)
    
    agent_name, _, _, agent_pos = agent
    agent_channel[agent_pos] = 1
    
    for coin in coins:
        field_channel[coin] = 2
    
    for other in others:
        _, _, _, other_pos = other
        others_channel[other_pos] = 1
    
    for bomb in bombs:
        bomb_pos, bomb_timer = bomb
        flag = 1
        if agent_pos == bomb_pos:
            flag = -1
        x, y = bomb_pos
        bombs_channel[x, y] = bomb_timer
        
        # Define the blast radius
        blast_radius = 3
        
        # Mark danger zone in all four directions
        for dx in range(1, blast_radius + 1):
            if x + dx < nrows and field[x + dx, y] != -1:
                bombs_channel[x + dx, y] = bomb_timer / BOMB_TIMER * flag
                if field[x + dx, y] == -1:  # Stop if obstacle is hit
                    break
            else:
                break

        for dx in range(1, blast_radius + 1):
            if x - dx >= 0 and field[x - dx, y] != -1:
                bombs_channel[x - dx, y] = bomb_timer / BOMB_TIMER * flag
                if field[x - dx, y] == -1:  # Stop if obstacle is hit
                    break
            else:
                break
        
        for dy in range(1, blast_radius + 1):
            if y + dy < ncols and field[x, y + dy] != -1:
                bombs_channel[x, y + dy] = bomb_timer / BOMB_TIMER * flag
                if field[x, y + dy] == -1:  # Stop if obstacle is hit
                    break
            else:
                break

        for dy in range(1, blast_radius + 1):
            if y - dy >= 0 and field[x, y - dy] != -1:
                bombs_channel[x, y - dy] = bomb_timer / BOMB_TIMER * flag
                if field[x, y - dy] == -1:  # Stop if obstacle is hit
                    break
            else:
                break
    
    # Combine all channels into a single feature vector
    stacked_channels = np.stack([field_channel, agent_channel, others_channel, bombs_channel], axis=2).flatten()
    stacked_channels = stacked_channels.reshape((34, 34))
    feature_tensor = torch.tensor(stacked_channels, dtype=torch.float32)
    return feature_tensor
