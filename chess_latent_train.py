from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import numpy as np

import chess.svg

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


all_numerical_positions = np.load('x_data.npy')
cutoff = int(np.floor(len(all_numerical_positions) * 0.8))

x_train = all_numerical_positions[:cutoff]
x_test = all_numerical_positions[cutoff:]


# The encoding process
input_img = Input(shape=(8, 8, 1))

############
# Encoding #
############

x = Conv2D(16, (1, 1), activation='linear', padding='same')(input_img)
x = MaxPooling2D((1, 1), padding='same')(x)
x = Conv2D(8, (2, 2), activation='linear', padding='same')(x)
x = MaxPooling2D((1, 1), padding='same')(x)
x = Conv2D(8, (2, 2), activation='linear', padding='same')(x)
encoded = MaxPooling2D((1, 1), padding='same')(x)

############
# Decoding #
############
x = Conv2D(8, (2, 2), activation='linear', padding='same')(encoded)
x = UpSampling2D((1, 1))(x)
x = Conv2D(8, (2, 2), activation='linear', padding='same')(x)
x = UpSampling2D((1, 1))(x)
x = Conv2D(16, (1, 1), activation='linear')(x)
x = UpSampling2D((1, 1))(x)

decoded = Conv2D(1, (1, 1), activation='linear', padding='same')(x)

# Declare the model
encoder = Model(input_img, encoded)
autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer='adadelta', loss='mse')

autoencoder.fit(x_train, x_train,
                epochs=1,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test)
                )

predictions = autoencoder.predict(x_test)
predictions_labels = np.trunc(predictions[:, :, :, :]).astype(int)
predictions_labels = np.vectorize(dictionary_of_numerical_positions.get)(predictions_labels)

for position_index, pos in enumerate(predictions_labels):
    label_board = chess.Board()
    label_board.clear()
    true_label_board = chess.Board()
    true_label_board.clear()

    for i, row in enumerate(pos):
        for j, piece in enumerate(row):
            if piece != '.':
                label_board.set_piece_at(i * 8 + j, chess.Piece.from_symbol(piece[0]))
            if dictionary_of_numerical_positions[x_test[position_index, i, j][0]] != '.':
                true_label_board.set_piece_at(i * 8 + j, chess.Piece.from_symbol(dictionary_of_numerical_positions[x_test[position_index, i, j][0]]))

    if label_board.__str__() == true_label_board.__str__():
        print("Prediction - WE WERE 100% right")
        print(label_board)
        print("Ground truth")
        print(true_label_board)
