import numpy as np

import chess.pgn

files = ['twic1305.pgn', 'twic1306.pgn', 'twic1307.pgn', 'twic1308.pgn']

# We will store all positions here
all_numerical_positions = []

number_to_piece = {
    0: 'r',  # lower case for white pieces
    1: 'n',
    2: 'b',
    3: 'q',
    4: 'k',
    5: 'p',
    6: '.',  # unoccupied square
    7: 'P',  # capital letters for black pieces
    8: 'R',
    9: 'N',
    10: 'B',
    11: 'Q',
    12: 'K',
}

# Reverse dictionary, given a piece 'K' return 12
piece_to_number = inv_map = {v: k for k, v in number_to_piece.items()}


for f in files:
    pgn = open("pgn_files/%s" % f)
    g = chess.pgn.read_game(pgn)

    while g is not None:
        board = g.board()
        board.reset()

        for move in g.mainline_moves():
            board.push(move)

            a = board.__str__()
            a = a.replace('\n', '').replace(' ', '')

            numerical_position = []
            for piece in a:
                numerical_position.append([piece_to_number[piece]])

            all_numerical_positions.append(np.array(numerical_position))

        g = chess.pgn.read_game(pgn)

all_numerical_positions = np.array(all_numerical_positions)
all_numerical_positions = all_numerical_positions.reshape((-1, 8, 8, 1))

np.save('x_data.npy', all_numerical_positions)
