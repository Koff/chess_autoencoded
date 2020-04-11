import numpy as np
import keras
import os

from keras import Model
from keras.layers import Input, Conv2D, Dense, Activation

EPOCHS = 10
BATCH_SIZE = 4096
MIDDLE_DIMENSIONS = 64
test = False

# Load chess positions data
all_numerical_positions = np.load('x_data.npy')


# If testing, remove data to cut training time
if test:
    all_numerical_positions = all_numerical_positions[:int(np.floor(all_numerical_positions.shape[0] / 20)), :, :, :]


all_numerical_positions = all_numerical_positions.reshape((-1, 8, 8, 1))


# Input into encoder
input_pos = Input(shape=(8, 8, 1, ))

# "encoded" is the encoded representation of the input
encoded = Dense(300)(input_pos)
encoded = Dense(200)(encoded)
encoded = Dense(150)(encoded)

# Bottle neck
encoded = Dense(MIDDLE_DIMENSIONS, activation='linear')(encoded)

# Decoder input
decoded_input = Input(shape=(MIDDLE_DIMENSIONS,))

# Rest of decoder
decoded = Dense(150)(decoded_input)
decoded = Dense(200)(decoded)
decoded = Dense(300)(decoded)
decoded = Dense(1)(decoded)

encoder = Model(input_pos, encoded, name='encoder')
encoder.summary()

decoder = Model(decoded_input, decoded, name='decoder')
decoder.summary()

# Variational autoencoder definition
outputs = decoder(encoder(input_pos))
vae = keras.Model(input_pos, outputs, name='vae_mlp')

vae.summary()
vae.compile(optimizer='adam', loss='mean_squared_error')

vae.fit(all_numerical_positions,
        all_numerical_positions,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1)

try:
    os.mkdir(f'models/middle_{MIDDLE_DIMENSIONS}')
except FileExistsError:
    print('Directory already exists.')

vae.save(f'models/middle_{MIDDLE_DIMENSIONS}/vae_all.h5')
encoder.save(f'models/middle_{MIDDLE_DIMENSIONS}/vae_only_encoder.h5')
decoder.save(f'models/middle_{MIDDLE_DIMENSIONS}/vae_only_decoder.h5')
