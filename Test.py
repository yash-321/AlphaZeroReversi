
from Arena import Arena
from MCTS import MCTS
from ReversiGame import ReversiGame
from ReversiPlayers import *
from NNetWrapper import NNetWrapper as NNet


import numpy as np
from utils import *


g = ReversiGame(8)

args = dotdict({'numMCTSSims': 25, 'cpuct':1.0})

n1 = NNet(g)
n1.load_checkpoint('./checkpoints/', 'best.pth.tar')
nmcts = MCTS(g, n1, args)
player1 = lambda x: np.argmax(nmcts.getActionProb(x, temp=0))

n2 = NNet(g)
n2.load_checkpoint('./checkpoints/', 'it5_8x8reversi.pth.tar')
n2mcts = MCTS(g, n2, args)
player2 = lambda x: np.argmax(n2mcts.getActionProb(x, temp=0))

#player2 = RandomPlayer(g).play
player2 = MinimaxPlayer(g, 5).play
#player2 = HumanReversiPlayer(g).play

arena = Arena(player1, player2, g, display=ReversiGame.display)
print(arena.playGames(10))
