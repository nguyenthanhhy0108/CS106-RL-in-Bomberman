import numpy as np
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# class for network
class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, kernel_size=3, bias=False)
        self.bn3 = nn.BatchNorm2d(24)

        def conv2d_size_out(size, kernel_size = 3, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 24
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization.
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

def custom_events(self, old_game_state: dict, new_game_state: dict, events: List[str]):
    if old_game_state == None:
        return []
    custom_events_list = []
    # give pos. reward for going towards coin and neg. for away
    # the distance is not the euclidian norm but x+y because we cannot moce diagonally
    # this is still not perfect bc of the walls but better than nothing
    # using try as an easy way to prevent old_game_state=None or no coins at all (latter not sure?)
    new_own_location = np.asarray(new_game_state['self'][-1])
    old_own_location = np.asarray(old_game_state['self'][-1])

    old_clostest_coin_indx = False
    old_clostest_coin_dist = 1000
    try:
        for i, old_coins in enumerate(old_game_state['coins']):
            dummy_dist = sum(abs(old_coins - old_own_location))

            if dummy_dist < old_clostest_coin_dist:
                old_clostest_coin_indx = i
                old_clostest_coin_dist = dummy_dist
        if old_clostest_coin_indx:
            if old_clostest_coin_dist < 8:
                if old_game_state['coins'][old_clostest_coin_indx] in new_game_state['coins']:
                    new_clostest_coin_dist = sum(abs(old_game_state['coins'][old_clostest_coin_indx] - new_own_location))

                    if new_clostest_coin_dist < old_clostest_coin_dist:
                        custom_events_list.append('MOVED_TO_COIN')
                    elif new_clostest_coin_dist > old_clostest_coin_dist:
                        custom_events_list.append('MOVED_AWAY_FROM_COIN')
    except Exception as e:
        print(e)

    try:
        #####################
        # rewards for bombs #
        for bomb in old_game_state['bombs']:
            bomb_location = bomb[0]
            bomb_timer = bomb[1]

            old_dist_bomb = np.sum(np.abs(bomb_location - old_own_location))
            new_dist_bomb = np.sum(np.abs(bomb_location - new_own_location))
            if old_dist_bomb < 5:
                if new_dist_bomb > old_dist_bomb:
                    custom_events_list.append('MOVED_AWAY_FROM_BOMB')
                else:
                    custom_events_list.append('MOVED_TO_BOMB')

                # smart dodge
                if bomb_timer > 0:
                    game_board = np.copy(old_game_state['field'])
                    game_board[game_board==1] = 0
                    game_board = np.abs(game_board)
                    explosion_sites = []
                    explosion_sites.append(bomb_location)
                    for direction in np.array([[0,1], [1,0], [-1,0], [0,-1]]):
                        # bomb radius is 3
                        for rad in range(1, 4):
                            # check if adjacent field is free
                            next_field = tuple(bomb_location+direction*rad)
                            if not game_board[next_field]:
                                explosion_sites.append(next_field)
                            else:
                                break

                    if tuple(old_own_location) in explosion_sites and tuple(new_own_location) not in explosion_sites:
                        custom_events_list.append('DODGED_BOMB')
                    if tuple(old_own_location) not in explosion_sites and tuple(new_own_location) in explosion_sites:
                        custom_events_list.append('ANTIDODGED_BOMB')

            #############
            # dead ends #
            if tuple(old_own_location)==bomb_location:
                chosen_direction = new_own_location - old_own_location
                if np.sum(chosen_direction) != 0:
                    free_tiles = [(x, y) for (x, y) in np.asarray(np.where(old_game_state['field']==0)).T]
                    turn_direction = chosen_direction[::-1]

                    loc_dummy = np.copy(new_own_location)
                    dead_end = True
                    for i in range(3):
                        if tuple(loc_dummy+turn_direction) in free_tiles or tuple(loc_dummy-turn_direction) in free_tiles:
                            dead_end = False
                            break
                        else:
                            loc_dummy = loc_dummy + chosen_direction
                    if dead_end:
                        custom_events_list.append('DEAD_END')
                    else:
                        custom_events_list.append('NOT_DEAD_END')


        ##############				
        # edge bombs #
        if 'BOMB_DROPPED' in events:
            dropped_bomb_location = tuple(old_own_location)
            if dropped_bomb_location in [(1,1), (1,15), (15,1), (15,15)]:
                custom_events_list.append('EDGE_BOMB')

            ####################
            # unnecessary bomb #
            game_board = np.copy(old_game_state['field'])
            game_board[game_board==1] = 0
            game_board = np.abs(game_board)
            explosion_sites = []
            explosion_sites.append(dropped_bomb_location)
            for direction in np.array([[0,1], [1,0], [-1,0], [0,-1]]):
                # bomb radius is 3
                for rad in range(1, 4):
                    # check if adjacent field is free
                    next_field = tuple(dropped_bomb_location+direction*rad)
                    if not game_board[next_field]:
                        explosion_sites.append(next_field)
                    else:
                        break
            #interesting targets
            crate_locations = np.asarray(np.where(old_game_state['field']==1)).T # all crate locations
            crate_tuples = [(xc, yc) for xc, yc in crate_locations]

            enemies = old_game_state['others']
            enemy_radius = np.array(np.meshgrid(np.arange(-3,4), np.arange(-3, 4))).T.reshape(-1, 2)
            if enemies:
                # also append opponents location to interesting targets
                for en in enemies:
                    crate_tuples.extend(([(x, y) for (x, y) in en[-1]+enemy_radius if 0<x<16 and 0<y<16 and x%2+y%2!=0]))

            smart_bomb = False
            for exp_site in explosion_sites:
                if exp_site in crate_tuples:
                    smart_bomb = True
            if smart_bomb:
                custom_events_list.append('SMART_BOMB')
            else:
                custom_events_list.append('DUMB_BOMB')


    except Exception as e:
        print(e)

    return custom_events_list