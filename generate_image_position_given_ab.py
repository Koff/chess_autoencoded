import numpy as np
import chess
from keras.models import load_model
from keras import backend as K
from matplotlib import pyplot as plt

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

decoder = load_model('models/vae_only_decoder.h5')
encoder = load_model('models/vae_only_encoder.h5')

# Load chess positions data
all_numerical_positions = np.load('x_data.npy')
m = 4
all_numerical_positions = (((all_numerical_positions[:, None] & (1 << np.arange(m)))) > 0).astype(int)
all_numerical_positions = all_numerical_positions.reshape((-1, 8, 8, 4))


def print_position_for_input_ab(input_a=0):
    input_a = decoder.predict(np.array([input_a]).reshape((-1, 1, 1, 1)))

    input_a = input_a.reshape((64, 4))
    input_a = np.where(input_a < 0.5, 0, 1)
    input_a = input_a.dot(1 << np.arange(input_a.shape[-1] - 1, -1, -1))

    b = chess.Board()

    for i, piece in enumerate(input_a):
        if piece != 6 and piece in dictionary_of_numerical_positions:
            b.set_piece_at(i, chess.Piece.from_symbol(dictionary_of_numerical_positions[piece]))

    print(b)


def print_image_for_position_and_estimates(pos):
    fig, axs = plt.subplots(2)

    original_position = pos.reshape(8, 32)
    plt.imshow(original_position, interpolation='nearest')

    axs[0].imshow(original_position, interpolation='nearest')
    predicted_position = np.round(decoder.predict(encoder.predict(pos.reshape(1, 8, 8, 4))))
    predicted_position = predicted_position.reshape(8, 32)
    axs[1].imshow(predicted_position, interpolation='nearest')
    plt.show()


print_image_for_position_and_estimates(all_numerical_positions[100])

a = encoder.predict(all_numerical_positions.reshape((-1, 8, 8, 4)))
a = a.reshape(-1, 1)

plt.scatter(a[:, 0], np.random.rand(1, len(a)), alpha=0.2, s=0.1)
plt.show()
