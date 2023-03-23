import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReversiNNet(nn.Module):
    def __init__(self, game, args):
        super(ReversiNNet, self).__init__()

        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Define the convolutional layer
        self.conv1 = nn.Conv2d(1, self.args.num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.args.num_channels)

        # Define the residual layers
        self.residual_layers = nn.ModuleList([ResidualLayer(self.args.num_channels) for _ in range(self.args.num_res_blocks)])

        # Define Hidden layers
        self.fc1 = nn.Linear(args.num_channels*self.board_x*self.board_y, 512)
        self.fc_bn1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.fc_bn2 = nn.BatchNorm1d(256)

        # Define policy and value head
        self.policy_fc = nn.Linear(256, self.action_size)
        self.value_fc = nn.Linear(256, 1)

    def forward(self, x):
        x = x.view(-1, 1, self.board_x, self.board_y)

        # Pass input through convolutional layer
        x = F.relu(self.bn1(self.conv1(x)))

        # Pass input through residual layers
        for layer in self.residual_layers:
            x = layer(x)

        x = x.view(-1, self.args.num_channels*self.board_x*self.board_y)

        # Pass through hidden layers
        x = F.dropout(F.relu(self.fc_bn1(self.fc1(x))), p=self.args.dropout, training=self.training)
        x = F.dropout(F.relu(self.fc_bn2(self.fc2(x))), p=self.args.dropout, training=self.training)

        policy = self.policy_fc(x)
        value = self.value_fc(x)
        
        return F.log_softmax(policy, dim=1), torch.tanh(value)


class ResidualLayer(nn.Module):
    def __init__(self, channels):
        super(ResidualLayer, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        # Pass input through first convolutional layer
        out = F.relu(self.bn1(self.conv1(x)))

        # Pass output through second convolutional layer
        out = self.bn2(self.conv2(out))

        # Add input to output and pass through activation function
        out = F.relu(out + x)

        return out
