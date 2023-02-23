import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ReversiNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(ReversiNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(args.num_channels)

        self.resBlocks = []
        for _ in range(args.num_res_blocks):
            res = ResBlock(args.num_channels)
            if self.args.cuda:
                res.cuda()
            self.resBlocks.append(res)

        self.fc1 = nn.Linear(args.num_channels*(self.board_x)*(self.board_y), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # Convolutional Layer

        for res in self.resBlocks:                                   # Residual Layers
            s = res(s)
        
        s = s.view(-1, self.args.num_channels*(self.board_x)*(self.board_y))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)                                            # Policy Head
        v = self.fc4(s)                                             # Value Head

        return F.log_softmax(pi, dim=1), torch.tanh(v)


    """
    Define an nn.Module class for a simple residual block with equal dimensions
    """
class ResBlock(nn.Module):

    """
    Iniialize a residual block with two convolutions followed by batchnorm layers
    """
    def __init__(self, num_channels:int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(num_channels)
        self.batchnorm2 = nn.BatchNorm2d(num_channels)

    def convblock(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        return x

    """
    Combine output with the original input
    """
    def forward(self, x): return x + self.convblock(x) # skip connection


