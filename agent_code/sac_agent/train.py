import pickle
import random
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import state_to_features
from .myfuncs import *

# new
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import warnings

import sys

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters
TRANSITION_HISTORY_SIZE = 500000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
RANDOM_BATCH_SIZE = 64
GAMMA = 0.99
TAU = 5e-4 # update rate for target netwsork
LEARNING_RATE = 1e-3 #4e-4
UPDATE_EVERY = 10 # training frequency in units of game steps


ActionDict = {'UP':0, 'RIGHT':1, 'DOWN':2, 'LEFT':3, 'WAIT':4, 'BOMB':5}

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
MAX_BUFFER_SIZE = 1e6

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    state_dim = 34 * 34  # Adjust these dimensions based on your environment
    action_dim = 6
    hidden_size = 256

    self.save_index = 0  # Initialize save_index here

    self.policy_net = SAC(state_dim, action_dim, hidden_size)
    self.target_net = SAC(state_dim, action_dim, hidden_size)
    # Synchronize the target network
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()

    # Optimizers
    self.optimizer_critic1 = optim.Adam(self.policy_net.critic1.parameters(), lr=LEARNING_RATE)
    self.optimizer_critic2 = optim.Adam(self.policy_net.critic2.parameters(), lr=LEARNING_RATE)
    self.optimizer_actor = optim.Adam(self.policy_net.actor.parameters(), lr=LEARNING_RATE)
    self.optimizer_alpha = optim.Adam([self.policy_net.log_alpha], lr=LEARNING_RATE)

    # Initialize self.optimizer
    self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE) # Replace YourOptimizer with the optimizer you are using

    # Learning rate schedulers
    self.scheduler_critic1 = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_critic1, T_max=1000)
    self.scheduler_critic2 = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_critic2, T_max=1000)
    self.scheduler_actor = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_actor, T_max=1000)

    # Replay buffer
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    # Training state
    self.total_steps = 0
    self.epoch = 0
    self.loss_list = []
    self.time_list = []
    self.steps_list = []
    self.points_list = []
    self.exploration_rate = 1.0
    self.inv_num = 0
    self.all_actions = np.zeros(6)
    self.target_entropy = -np.prod((self.policy_net.actor[-1].out_features,)).item()  # or another value suitable for your action space

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    try:
        custom_events_list = custom_events(old_game_state, new_game_state, events)
        events.extend(custom_events_list)

        self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

        if old_game_state is not None:
            self.transitions.append(Transition(state_to_features(old_game_state), ActionDict[self_action], state_to_features(new_game_state), reward_from_events(self, events)))

        # Limit the size of transitions to prevent memory overflow
        if len(self.transitions) > MAX_BUFFER_SIZE:
            self.transitions.pop(0)

        # Optimization step
        if self.total_steps >= 20000 and self.total_steps % UPDATE_EVERY == 0:
            # Save model every 10000 steps
            if self.total_steps % 10000 == 0:
                save_model(self)

            # Sample random batch from transitions
            random_indices = np.random.permutation(np.arange(len(self.transitions)))[:RANDOM_BATCH_SIZE]
            random_batch = [self.transitions[i] for i in random_indices]
            batch = Transition(*zip(*random_batch))

            state_batch = torch.cat(batch.state).view(-1, 34 * 34)  # Adjust the view dimensions
            action_batch = torch.tensor(batch.action, dtype=torch.float32)
            reward_batch = torch.tensor(batch.reward, dtype=torch.float32)
            next_state_batch = torch.cat([s for s in batch.next_state if s is not None]).view(-1, 34 * 34)  # Adjust the view dimensions

            # Critic update
            with torch.no_grad():
                next_state_Q1 = self.target_net.Q1(next_state_batch, self.target_net.actor(next_state_batch))
                next_state_Q2 = self.target_net.Q2(next_state_batch, self.target_net.actor(next_state_batch))
                next_state_Q = torch.min(next_state_Q1, next_state_Q2)
                next_state_value = torch.zeros_like(next_state_Q.max(1)[0].detach())
                next_state_value[:] = next_state_Q.max(1)[0].detach()

            expected_state_action_value = reward_batch.unsqueeze(1) + GAMMA * next_state_value

            state_action_value = self.policy_net(state_batch).gather(1, action_batch.long().unsqueeze(1))

            warnings.filterwarnings("ignore", message="Using a target size")

            critic_loss = F.mse_loss(state_action_value, expected_state_action_value)

            # Optimize critic
            self.optimizer_critic1.zero_grad()
            self.optimizer_critic2.zero_grad()
            critic_loss.backward()
            self.optimizer_critic1.step()
            self.optimizer_critic2.step()

            # Actor update
            actions = self.policy_net.actor(state_batch)
            log_pi = self.policy_net.entropy_term(actions)
            Q1 = self.policy_net.Q1(state_batch, actions)
            Q2 = self.policy_net.Q2(state_batch, actions)
            Q = torch.min(Q1, Q2)

            actor_loss = (self.policy_net.alpha() * log_pi - Q).mean()

            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            # Entropy temperature update
            alpha_loss = -(self.policy_net.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.optimizer_alpha.zero_grad()
            alpha_loss.backward()
            self.optimizer_alpha.step()

            self.policy_net.log_alpha.data.copy_(torch.clamp(self.policy_net.log_alpha.data, min=-10, max=2))

            # Soft target update
            for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

            # Append losses to lists for tracking
            self.loss_list.append((critic_loss.item(), actor_loss.item(), alpha_loss.item()))
            self.time_list.append(self.total_steps)

            self.scheduler_critic1.step()
            self.scheduler_critic2.step()
            self.scheduler_actor.step()

        self.total_steps += 1

    except Exception as e:
        self.logger.error(f"An error occurred: {e}")
        raise

def save_model(self):
    '''
    When called, this function saves the current model.
    '''

    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    torch.save({
            'epoch': self.epoch,
            'total_steps':self.total_steps,
            'save_index': self.save_index,
            'model_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }, 'my-saved-model.tar')

    #create checkpoints once in a while
    if self.total_steps%50000 == 0:
        torch.save({
        'epoch': self.epoch,
        'total_steps':self.total_steps,
        'save_index': self.save_index,
        'model_state_dict': self.policy_net.state_dict(),
        'target_state_dict': self.target_net.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        }, 'checkpoints/my-saved-model{}.tar'.format(self.save_index))

        with open("transition_history.pt", "wb") as file:
            pickle.dump(self.transitions, file)

        print('checkpoint {} created!'.format(self.save_index))

    if not os.path.exists('metrics'):
        os.makedirs('metrics')

    # save metrics for analysis
    np.save('metrics/loss{}'.format(self.save_index), np.array(self.loss_list))
    np.save('metrics/time{}'.format(self.save_index), np.array(self.time_list))
    np.save('metrics/score{}'.format(self.save_index), np.array(self.points_list))
    np.save('metrics/steps{}'.format(self.save_index), np.array(self.steps_list))
    self.loss_list = []
    self.time_list = []
    self.steps_list = []
    self.points_list = []


    # print points
    print('Saved model and metrics', self.save_index)
    self.save_index += 1

    # decrease exploration rate after each game
    self.exploration_rate = max(0.1, self.exploration_rate*0.95)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.
    """
    # Lưu số bước và điểm cuối cùng của ván
    self.steps_list.append(last_game_state['step'])
    self.points_list.append(last_game_state['self'][1])

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Lưu transition cuối cùng của ván
    self.transitions.append(Transition(state_to_features(last_game_state), ActionDict[last_action], None, reward_from_events(self, events)))

    # In thông tin sau mỗi 1000 epoch
    if self.epoch % 1000 == 0:
        print(self.all_actions, 'inv', self.inv_num, '\tpoints',
            last_game_state['self'][1], 'expl', np.round(self.exploration_rate, 3), '\ttotal steps', self.total_steps, '\tepoch', self.epoch)

    # Reset các giá trị đếm
    self.inv_num = 0
    self.all_actions = np.zeros(6)
    self.epoch += 1

game_rewards = {
        e.KILLED_OPPONENT: 30,
        e.COIN_COLLECTED: 30,
        e.MOVED_TO_COIN: 5,
        e.MOVED_AWAY_FROM_BOMB: 2,
        e.DODGED_BOMB: 3,
        e.CRATE_DESTROYED: 30,
        e.SMART_BOMB: 6,
        e.BOMB_DROPPED: 5,
        e.NOT_DEAD_END: .5,
        ########
        e.DEAD_END: -.5,
        e.WAITED: -30,
        e.INVALID_ACTION: -2,
        e.DUMB_BOMB: -10,
        e.MOVED_AWAY_FROM_COIN: -1,
        e.GOT_KILLED: -30,
        e.MOVED_TO_BOMB: -3,
        e.ANTIDODGED_BOMB: -5,
        e.EDGE_BOMB: -30
    }
for key in game_rewards:
    game_rewards[key] = game_rewards[key]/30

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
