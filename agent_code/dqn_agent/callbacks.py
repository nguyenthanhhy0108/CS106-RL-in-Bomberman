import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from scipy.special import softmax

from .myfuncs import DQN

from collections import namedtuple, deque
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


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
    ##### training ######
    self.resume_training = False

    self.model = DQN(15, 15, 6)
    if self.train or not os.path.isfile("my-saved-model.tar"):

        ###### start a new model ######
        if not os.path.isfile("my-saved-model.tar"):
            self.logger.info("Setting up model from scratch.")
            self.epoch = 0
            self.save_index = 0
            print('new model')

        ##### continue training #######
        else:
            print('loading model for further training')
            self.resume_training = True
            self.logger.info("Train existing model.")
            checkpoint = torch.load('my-saved-model.tar')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.train()
            #self.transitions = checkpoint['transition_history']
            self.optimizer_dummy = checkpoint['optimizer_state_dict']
            self.total_steps = checkpoint['total_steps']
            self.epoch = checkpoint['epoch']

            self.target_net_dummy = checkpoint['target_state_dict']
            self.transitions = deque(maxlen=500000)
            try:
                with open("transition_history.pt", "rb") as file:
                    self.transitions.extend(pickle.load(file))
                print('successfully loaded old transitions')
            except:
                print('creating new, empty transitions')
                self.total_steps = 0

            try:
                self.save_index = checkpoint['save_index']
                print('save index is at', self.save_index)
            except:
                print('could not find save index. Starting at zero again.')
                self.save_index = 0
    ##### gaming #####
    else:
        self.logger.info("Loading model from saved state.")
        checkpoint = torch.load('my-saved-model.tar')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print('load model for gaming')
    self.inv_num = 0


def act(self, game_state: dict) -> str:
    if game_state['step'] == 1:
        self.init_pos = game_state['self'][-1]
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    self.logger.debug("Querying model for action.")

    if self.train:
        self.total_steps += 1
        # return a random action
        if random.random() < self.exploration_rate:
            self.logger.debug("Choosing action purely at random.")
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .05, .15])#p=[.22, .22, .22, .22, .11, 0.01])
        # make policy net choose an action
        else:
            with torch.no_grad():
                myprobs = self.policy_net(state_to_features(game_state).view(1,1,15,15)).detach().numpy()[0]
            self.all_actions[np.argmax(myprobs)] += 1
            self.Q_list.append(np.max(myprobs))
            return ACTIONS[np.argmax(myprobs)]

    else:
        # choose only best action when not training
        with torch.no_grad():
            myprobs = self.model(state_to_features(game_state).view(1,1,15,15)).detach().numpy()[0]
        # myprobs = myprobs
        # print(myprobs)
        return ACTIONS[np.argmax(myprobs)] 
    
    


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    full_field = np.copy(game_state['field'])
    coin_locations = game_state['coins']
    full_field[game_state['self'][-1]] = 10
    
    for bomb in game_state['bombs']:
        full_field[bomb[0]] = -10+bomb[1]

    if coin_locations:
        full_field[tuple(np.asarray(coin_locations).T)] = 2

    enemies = game_state['others']
    if enemies:
        for en in enemies:
            full_field[en[-1]] = -5

    # now crop the outer borders bc they are always the same and normalize
    full_field = full_field[1:-1, 1:-1]
    full_field = full_field/10
    return torch.from_numpy(full_field).float()