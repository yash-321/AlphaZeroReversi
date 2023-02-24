
from Arena import Arena
from MCTS import MCTS
from reversi.ReversiGame import ReversiGame
from reversi.ReversiPlayers import *
from reversi.pytorch.NNet import NNetWrapper as NNet


import numpy as np
from utils import *


g = ReversiGame(8)

args = dotdict({'numMCTSSims': 50, 'cpuct':1.0})

n1 = NNet(g)
pmcts = MCTS(g, n1, args)
player2 = lambda x: np.argmax(pmcts.getActionProb(x, temp=0))
n2 = NNet(g)
#n2.load_checkpoint('./checkpoints/', 'best.pth.tar')
nmcts = MCTS(g, n2, args)
player1 = lambda x: np.argmax(nmcts.getActionProb(x, temp=0))

p1 = HumanReversiPlayer(g).play
p2 = HumanReversiPlayer(g).play

player1 = RandomPlayer(g).play

# arena = Arena(p1, p2, g, ReversiGame.display)

# print(arena.playGame(True))

arena = Arena(player1, player2, g, display=ReversiGame.display)
print(arena.playGames(5))