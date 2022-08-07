import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
from utils import *


class RandomAgent():
    """
    an agent take random actions
    """
    def __init__(self, num_actions=4):
        self.num_actions = num_actions
    def get_action(self):
        action = random.randint(0, self.num_actions - 1)
        return action

class DQN(nn.Module):
    def __init__(self, num_actions):
        """
        :param num_actions: the number of valid actions
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, 8, stride=4)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.l1 = nn.Linear(2592, 256)
        self.l2 = nn.Linear(256, num_actions)

    def forward(self, x):
        """
        :param x: state, shape:[batch_size, 4, 84,84]
        :return: Q scores, shape:[batch_size, num_actions]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        #shape of x: [batch_size, 32, 9, 9]
        x = x.view(x.size(0),-1)
        #shape of x: [batch_size, 2592]
        x = F.relu(self.l1(x))
        #shape of x: [batch_size, 256]
        Q = self.l2(x)
        #shape of Q: [batch_size, num_actions]
        return Q


class ReplayBuffer():
    """Replay Buffer stores the last N transitions"""

    def __init__(self, max_size=1000, history=4, batch_size=32):
        """
        args:
          max_size: the maximal number of stored transitions
          history: the number of frames stacked to create a state
          batch_size: the number of transitions returned in a minibatch
        """
        self.max_size = max_size
        self.history = history
        self.batch_size = batch_size
        self.frames = deque([], maxlen=max_size)
        self.actions = deque([], maxlen=max_size)
        self.next_frames = deque([], maxlen=max_size)
        self.rewards = deque([], maxlen=max_size)
        self.is_dones = deque([], maxlen=max_size)
        self.indices = [None] * batch_size

    def add_experience(self, frame, action, next_frame, reward, is_done):
        """
        form of data: (frame, action, next_frame, reward, is_done)
        frame: torch tensor, shape:[n_channels,width,height]
        action: interger
        next_frame: torch tensor, shape:[n_channels=1, width, height]
        is_done: whether the next frame is a terminal state
        """
        self.frames.append(frame)
        self.actions.append(action)
        self.next_frames.append(next_frame)
        self.rewards.append(reward)
        self.is_dones.append(is_done)

    def get_current_state(self):
        """create current state if there exists at least 4 non-terminal frames"""
        state = list(self.next_frames)[-self.history:]
        state = torch.cat(state, dim=0)  # shape:[n_timesteps, width, height]
        return state

    def get_valid_indices(self):
        experience_size = len(self.frames)
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.history, experience_size - 1)
                if True in list(self.is_dones)[index - self.history:index]:
                    continue
                break
            self.indices[i] = index

    def get_minibatch(self):
        """
        Returns a minibatch
        """
        batch = []
        self.get_valid_indices()

        for idx in self.indices:
            state = list(self.frames)[idx - self.history + 1:idx + 1]
            state = torch.cat(state, dim=0)
            # shape:[n_timesteps, width, height]
            next_state = list(self.next_frames)[idx - self.history + 1:idx + 1]
            next_state = torch.cat(next_state, dim=0)
            # shape:[n_timesteps, width, height]
            action = self.actions[idx]
            reward = self.rewards[idx]
            is_done = self.is_dones[idx]

            batch.append((state, action, next_state, reward, is_done))

        return batch
















