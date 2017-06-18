# -*- coding: utf-8 -*-

import numpy as np


class TicTacToeEnv:

    def __init__(self):
        self.board, self.current_player = None, None
        self.reset()

    def reset(self):
        self.board = np.zeros(9, dtype=np.int32)
        self.current_player = "x"

    def step(self, index):
        if self.board[index] != 0:
            print("Invalid move!!")
        elif self.current_player == "x":
            self.board[index] = 1
            self.current_player = "o"
        else:
            self.board[index] = -1
            self.current_player = "x"
        return np.array(self.board)

    def render(self):
        markers = []
        for i in self.board:
            if i == 0:
                markers.append("_")
            elif i == 1:
                markers.append("x")
            else:
                markers.append("o")
        print("{0}\t{1}\t{2}".format(markers[0], markers[1], markers[2]))
        print("{0}\t{1}\t{2}".format(markers[3], markers[4], markers[5]))
        print("{0}\t{1}\t{2}\n".format(markers[6], markers[7], markers[8]))
