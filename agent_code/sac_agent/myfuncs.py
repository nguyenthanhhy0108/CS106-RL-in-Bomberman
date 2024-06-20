import numpy as np
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += residual
        return F.relu(out)

class SAC(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(SAC, self).__init__()

        # Critic 1
        self.critic1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            ResidualBlock(hidden_size, hidden_size // 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1)
        )

        # Critic 2
        self.critic2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            ResidualBlock(hidden_size, hidden_size // 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1)
        )

        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            ResidualBlock(hidden_size, hidden_size // 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, action_dim)
        )

        self.log_alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, state):
        return self.actor(state)

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.critic1(sa)

    def Q2(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.critic2(sa)

    def alpha(self):
        return torch.exp(self.log_alpha)

    def entropy_term(self, actions):
        return -self.log_alpha * (actions.pow(2).log())

def custom_events(old_game_state: dict, new_game_state: dict, events: List[str]):
    custom_events_list = []

    if old_game_state is None:
        return custom_events_list

    try:
        old_own_location = np.asarray(old_game_state['self'][-1])
        new_own_location = np.asarray(new_game_state['self'][-1])

        # Check proximity to coins
        if 'coins' in old_game_state:
            old_closest_coin_indx = None
            old_closest_coin_dist = 1000
            for i, old_coin in enumerate(old_game_state['coins']):
                dummy_dist = sum(abs(old_coin - old_own_location))
                if dummy_dist < old_closest_coin_dist:
                    old_closest_coin_indx = i
                    old_closest_coin_dist = dummy_dist

            if old_closest_coin_indx is not None and old_closest_coin_dist < 8:
                if old_game_state['coins'][old_closest_coin_indx] in new_game_state['coins']:
                    new_closest_coin_dist = sum(abs(old_game_state['coins'][old_closest_coin_indx] - new_own_location))
                    if new_closest_coin_dist < old_closest_coin_dist:
                        custom_events_list.append('MOVED_TO_COIN')
                    elif new_closest_coin_dist > old_closest_coin_dist:
                        custom_events_list.append('MOVED_AWAY_FROM_COIN')

        # Check proximity to bombs
        if 'bombs' in old_game_state:
            for bomb in old_game_state['bombs']:
                bomb_location = bomb[0]
                old_dist_bomb = np.sum(np.abs(bomb_location - old_own_location))
                new_dist_bomb = np.sum(np.abs(bomb_location - new_own_location))
                if old_dist_bomb < 5:
                    if new_dist_bomb > old_dist_bomb:
                        custom_events_list.append('MOVED_AWAY_FROM_BOMB')
                    else:
                        custom_events_list.append('MOVED_TO_BOMB')
                    if bomb[1] > 0:
                        if tuple(old_own_location) in get_explosion_sites(old_game_state, bomb_location) \
                                and tuple(new_own_location) not in get_explosion_sites(old_game_state, bomb_location):
                            custom_events_list.append('DODGED_BOMB')
                        if tuple(old_own_location) not in get_explosion_sites(old_game_state, bomb_location) \
                                and tuple(new_own_location) in get_explosion_sites(old_game_state, bomb_location):
                            custom_events_list.append('ANTIDODGED_BOMB')

        # Check for edge bombs
        if 'BOMB_DROPPED' in events:
            if tuple(old_own_location) in [(1, 1), (1, 15), (15, 1), (15, 15)]:
                custom_events_list.append('EDGE_BOMB')

        # Check unnecessary bomb
        if 'BOMB_DROPPED' in events and 'field' in old_game_state:
            if tuple(old_own_location) in get_smart_bomb_targets(old_game_state):
                custom_events_list.append('SMART_BOMB')
            else:
                custom_events_list.append('DUMB_BOMB')

    except Exception as e:
        # print(e)
        pass

    return custom_events_list

def get_explosion_sites(game_state: dict, bomb_location: tuple):
    game_board = np.copy(game_state['field'])
    game_board[game_board == 1] = 0
    game_board = np.abs(game_board)
    explosion_sites = [bomb_location]
    for direction in np.array([[0, 1], [1, 0], [-1, 0], [0, -1]]):
        for rad in range(1, 4):
            next_field = tuple(bomb_location + direction * rad)
            if not game_board[next_field]:
                explosion_sites.append(next_field)
            else:
                break
    return explosion_sites

def get_smart_bomb_targets(game_state: dict):
    bomb_location = tuple(game_state['self'][-1])
    game_board = np.copy(game_state['field'])
    game_board[game_board == 1] = 0
    game_board = np.abs(game_board)
    explosion_sites = [bomb_location]
    for direction in np.array([[0, 1], [1, 0], [-1, 0], [0, -1]]):
        for rad in range(1, 4):
            next_field = tuple(bomb_location + direction * rad)
            if not game_board[next_field]:
                explosion_sites.append(next_field)
            else:
                break
    crate_locations = np.asarray(np.where(game_state['field'] == 1)).T
    crate_tuples = [(x, y) for x, y in crate_locations]
    enemies = game_state['others']
    enemy_radius = np.array(np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))).T.reshape(-1, 2)
    if enemies:
        for en in enemies:
            crate_tuples.extend(([(x, y) for (x, y) in en[-1] + enemy_radius if 0 < x < 16 and 0 < y < 16 and x % 2 + y % 2 != 0]))

    smart_bomb = False
    for exp_site in explosion_sites:
        if exp_site in crate_tuples:
            smart_bomb = True
    return smart_bomb