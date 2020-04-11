import numpy as np
import chess
import chess.svg
import matplotlib.image as mpimg

from keras.models import load_model
from keras import backend as K
from matplotlib import pyplot as plt

MIDDLE_DIMENSIONS = 4

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


K.clear_session()

decoder = load_model(f'models/middle_{MIDDLE_DIMENSIONS}/vae_only_decoder.h5')
encoder = load_model(f'models/middle_{MIDDLE_DIMENSIONS}/vae_only_encoder.h5')

# Load chess positions data
all_numerical_positions = np.load('x_data.npy')
all_numerical_positions = all_numerical_positions.reshape((-1, 64))


def create_board_with_position(position):
    pos = position.reshape(64)
    board = chess.Board()
    board.clear_board()
    for i, piece in enumerate(pos):
        if piece != 6 and piece in dictionary_of_numerical_positions:
            board.set_piece_at(i, chess.Piece.from_symbol(dictionary_of_numerical_positions[piece]))
    return board


def print_position_for_input_a(input_a=0):
    input_a = decoder.predict(np.array([input_a]).reshape((-1, 1, 1, 1)))

    input_a = input_a.reshape((64, 4))
    input_a = np.where(input_a < 0.5, 0, 1)
    input_a = input_a.dot(1 << np.arange(input_a.shape[-1] - 1, -1, -1))

    b = chess.Board()

    for i, piece in enumerate(input_a):
        if piece != 6 and piece in dictionary_of_numerical_positions:
            b.set_piece_at(i, chess.Piece.from_symbol(dictionary_of_numerical_positions[piece]))
    print(b)


def predict_position(position):
    predicted_position = np.round(decoder.predict(encoder.predict(position.reshape(1, 64))))
    input_a = predicted_position.reshape(64)
    return input_a


def print_image_for_position_and_estimates(pos):
    fig, axs = plt.subplots(3)

    original_position = pos.reshape(8, 32)

    axs[0].imshow(original_position, interpolation='nearest')
    predicted_position = predict_position(pos)
    predicted_position = predicted_position.reshape(8, 32)
    axs[1].imshow(predicted_position, interpolation='nearest')

    axs[2].imshow(predicted_position - original_position, interpolation='nearest')
    plt.axis('off')
    plt.show()


def render_boards():
    fig, axs = plt.subplots(2)

    img = mpimg.imread('images/board0.png')
    axs[0].imshow(img, interpolation='nearest')
    plt.axis('off')

    img = mpimg.imread('images/board1.png')
    axs[1].imshow(img, interpolation='nearest')
    plt.axis('off')

    plt.show()


position_index = 1660


predicted_position = predict_position(all_numerical_positions[position_index])

print("Real\n")
print(create_board_with_position(all_numerical_positions[position_index]))

print("\n\nPredicted\n")
print(create_board_with_position(predicted_position))
