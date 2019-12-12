from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model

import numpy as np

import chess.pgn
import chess.svg

pgn = open("twic1309.pgn")

all_numerical_positions = []

g = chess.pgn.read_game(pgn)

dictionary_of_numerical_positions = {
    0: 'r',
    1: 'n',
    2: 'b',
    3: 'q',
    4: 'k',
    5: 'p',
    6: '.',
    7: 'P',
    8: 'R',
    9: 'N',
    10: 'B',
    11: 'Q',
    12: 'K',
}
dictionary_of_positions = {
    'r': ('white_rook', 0),
    'n': ('white_knight', 1),
    'b': ('white_bishop', 2),
    'q': ('white_queen', 3),
    'k': ('white_king', 4),
    'p': ('white_pawn', 5),
    '.': ('empty', 6),
    'P': ('black_pawn', 7),
    'R': ('black_rook', 8),
    'N': ('black_knight', 9),
    'B': ('black_bishop', 10),
    'Q': ('black_queen', 11),
    'K': ('black_king', 12)
}

while g is not None:
    board = g.board()
    board.reset()
    for move in g.mainline_moves():
        board.push(move)

        a = board.__str__()
        a = a.replace('\n', '').replace(' ', '')

        numerical_position = []
        for piece in a:
            numerical_position.append([dictionary_of_positions[piece][1]])

        all_numerical_positions.append(np.array(numerical_position))

    g = chess.pgn.read_game(pgn)

all_numerical_positions = np.array(all_numerical_positions)
all_numerical_positions = all_numerical_positions.reshape((-1, 8, 8, 1))

np.save('x_data.npy', all_numerical_positions)

