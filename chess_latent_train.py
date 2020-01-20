import numpy as np
import chess.svg

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras.losses import mse, binary_crossentropy, categorical_crossentropy


# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


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
# Next line reduces the amount of data to 10% for faster training
# all_numerical_positions = all_numerical_positions[:int(np.floor(all_numerical_positions.shape[0]/5)), :, :, :]

cutoff = int(np.floor(len(all_numerical_positions) * 0.7))

x_train = all_numerical_positions[:cutoff]
x_test = all_numerical_positions[cutoff:]

image_size = 8
original_dim = image_size * image_size

x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])

input_shape = (original_dim, )
batch_size = 128
intermediate_dim = 128
latent_dim = 10
epochs = 5

# The encoding process

# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean', activation='relu')(x)
z_log_var = Dense(latent_dim, name='z_log_var', activation='relu')(x)


# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='softmax')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

models = (encoder, decoder)
data = (x_test, )

reconstruction_loss = categorical_crossentropy(inputs, outputs)

reconstruction_loss *= original_dim
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='nadam', metrics=['accuracy'])
vae.summary()

vae.fit(x_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))

predictions = vae.predict(x_test)
predictions_labels = np.trunc(predictions[:, :]).astype(int)
predictions_labels = np.vectorize(dictionary_of_numerical_positions.get)(predictions_labels)


for position_index, pos in enumerate(predictions_labels):
    label_board = chess.Board()
    label_board.clear()
    true_label_board = chess.Board()
    true_label_board.clear()

    for i, piece in enumerate(pos):
        if piece != '.' and piece is not None and piece != 'None':
            label_board.set_piece_at(i, chess.Piece.from_symbol(piece))
        if dictionary_of_numerical_positions[x_test[position_index, i]] != '.':
            true_label_board.set_piece_at(i, chess.Piece.from_symbol(
                dictionary_of_numerical_positions[x_test[position_index, i]]))

    if label_board.__str__() == true_label_board.__str__():
        print("Prediction - WE WERE 100% right")
        print(true_label_board)
