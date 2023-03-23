import numpy as np


class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanReversiPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print("[", int(i/self.game.n), int(i%self.game.n), end="] ")
        while True:
            input_move = input()
            input_a = input_move.split(" ")
            if len(input_a) == 2:
                try:
                    x,y = [int(i) for i in input_a]
                    if ((0 <= x) and (x < self.game.n) and (0 <= y) and (y < self.game.n)) or \
                            ((x == self.game.n) and (y == 0)):
                        a = self.game.n * x + y if x != -1 else self.game.n ** 2
                        if valid[a]:
                            break
                except ValueError:
                    # Input needs to be an integer
                    'Invalid integer'
            print('Invalid move')
        return a


class GreedyReversiPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a]==0:
                continue
            nextBoard, _ = self.game.getNextState(board, 1, a)
            score = self.game.getScore(nextBoard, 1)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]


class MinimaxPlayer():
    def __init__(self, game, depth):
        self.game = game
        self.depth = depth
        self.positionWeights = [
            [120, -20,  20,   5,   5,  20, -20, 120],
            [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
            [ 20,  -5,  15,   3,   3,  15,  -5,  20],
            [  5,  -5,   3,   3,   3,   3,  -5,   5],
            [  5,  -5,   3,   3,   3,   3,  -5,   5],
            [ 20,  -5,  15,   3,   3,  15,  -5,  20],
            [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
            [120, -20,  20,   5,   5,  20, -20, 120]
        ]

    def play(self, board):
        _, action = self.minimax(board, self.depth, -np.inf, np.inf, True)
        return action

    def minimax(self, board, depth, alpha, beta, maximizingPlayer):
        if depth == 0 or self.game.getGameEnded(board, 1) != 0:
            if maximizingPlayer:
                return self.getScore(board, 1), None
            return self.getScore(board, -1), None

        valid_moves = self.game.getValidMoves(board, 1)
        if maximizingPlayer:
            max_score = -np.inf
            max_action = None
            for action in range(self.game.getActionSize()):
                if valid_moves[action]:
                    next_board, _ = self.game.getNextState(board, 1, action)
                    score, _ = self.minimax(next_board, depth-1, alpha, beta, False)
                    if score > max_score:
                        max_score = score
                        max_action = action
                    alpha = max(alpha, score)
                    if alpha >= beta:
                        break
            return max_score, max_action
        else:
            min_score = np.inf
            min_action = None
            for action in range(self.game.getActionSize()):
                if valid_moves[action]:
                    next_board, _ = self.game.getNextState(board, -1, action)
                    score, _ = self.minimax(next_board, depth-1, alpha, beta, True)
                    if score < min_score:
                        min_score = score
                        min_action = action
                    beta = min(beta, score)
                    if beta <= alpha:
                        break
            return min_score, min_action

    def getScore(self, board, player):
        # Weighted heuristic evaluation function for board state
        opp_player = -player
        score = 0
        for i in range(self.game.n):
            for j in range(self.game.n):
                if board[i][j] == player:
                    score += self.positionWeights[i][j]
                elif board[i][j] == opp_player:
                    score -= self.positionWeights[i][j]
        return score

