import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
from utils import *


class RandomAgent():
    """
    an agent take random actions
    """
    def __init__(self, num_actions=4):
        self.num_actions = num_actions
    def get_action(self, state=None):
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

    def __init__(self, max_size=10000, history=4, batch_size=32):
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

    def current_state_available(self):
        """
        Check whether the current state is avalable
        """
        if (len(self.is_dones) < self.history) or (True in list(self.is_dones)[-self.history:]):
            return False
        else:
            return True

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


class QAgent():
    def __init__(self, num_actions, optimizer_type="Adam", lr=0.00025,
                 loss_criterion=nn.SmoothL1Loss(),
                 load_policy_path=None,
                 load_target_path=None,
                 trained_epochs=0):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(num_actions)
        self.target_net = DQN(num_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        if load_policy_path is not None:
            # load policy network
            self.policy_net = torch.load(load_policy_path)
        if load_target_path is not None:
            # load target network
            self.target_net = torch.load(load_target_path)

        self.policy_net.to(device)
        self.target_net.to(device)

        if optimizer_type.upper() == "ADAM":
            self.optimizer = optim.Adam(self.policy_net.parameters(),
                                        lr=lr)
        else:
            self.optimizer = optim.RMSprop(self.policy_net.parameters(),
                                           lr=lr)
        self.loss_criterion = loss_criterion
        self.trained_epochs = trained_epochs

    def get_action(self, state):
        """predict the action for a state"""
        # state, shape:[batch_size=1, 4, 84,84]
        with torch.no_grad():
            self.policy_net.eval()
            Q = self.policy_net(state.float())
            # Q, shape:[batch_size=1, num_actions]
            action = torch.argmax(Q, -1)
        return action.item()

    def replay(self, replay_buffer, gamma=0.99):
        # train the policy net one epoch
        batch = replay_buffer.get_minibatch()
        batch_trans = list(map(list, zip(*batch)))
        states = torch.stack(batch_trans[0], dim=0)
        actions = batch_trans[1]
        batch_size = len(actions)
        batch_indices = range(batch_size)
        next_states = torch.stack(batch_trans[2], dim=0)
        rewards = torch.tensor(batch_trans[3])
        is_dones = torch.tensor(batch_trans[4])
        # predict Q values
        Q_predicted = self.policy_net(states.float())
        # get predicted Q(s,a)
        Q_predicted = Q_predicted[batch_indices, actions]

        # get target Q values
        with torch.no_grad():
            self.target_net.eval()
            Q_next = self.target_net(next_states.float())
            Q_next_max = torch.max(Q_next, -1).values
            Q_next_max[is_dones] = 0.
            Q_target = gamma * Q_next_max + rewards

        # train policy network
        loss = self.loss_criterion(Q_predicted, Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.trained_epochs += 1

        return self.trained_epochs, loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
















