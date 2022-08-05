import torch
import torch.nn as nn
import torch.nn.functional as F

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



class ReplayMemory(object):
    pass