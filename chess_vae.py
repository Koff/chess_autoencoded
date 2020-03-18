import numpy as np
import keras

from keras import Model
from keras.layers import Input, Conv2D, UpSampling2D, AveragePooling2D, Dense, Dropout

EPOCHS = 30
BATCH_SIZE = 4096
MIDDLE_DIMENSIONS = 3
test = False

# Load chess positions data
all_numerical_positions = np.load('x_data.npy')


# If testing, remove data to cut training time
if test:
    all_numerical_positions = all_numerical_positions[:int(np.floor(all_numerical_positions.shape[0] / 20)), :, :, :]


all_numerical_positions = all_numerical_positions.reshape((-1, 64))


# Input into encoder
input_pos = Input(shape=(64,))

# "encoded" is the encoded representation of the input
encoded = Dense(100, activation='linear')(input_pos)
encoded = Dense(30, activation='linear')(encoded)
encoded = Dense(10, activation='linear')(encoded)

# Bottle neck
encoded = Dense(MIDDLE_DIMENSIONS, activation='linear')(encoded)

# Decoder input
decoded_input = Input(shape=(MIDDLE_DIMENSIONS,))

# Rest of decoder
decoded = Dense(10, activation='linear')(decoded_input)
decoded = Dense(30, activation='linear')(decoded)
decoded = Dense(100, activation='linear')(decoded)

decoded = Dense(64, activation='linear')(decoded)

encoder = Model(input_pos, encoded, name='encoder')
encoder.summary()

decoder = Model(decoded_input, decoded, name='decoder')
decoder.summary()

# Variational autoencoder definition
outputs = decoder(encoder(input_pos))
vae = keras.Model(input_pos, outputs, name='vae_mlp')

vae.summary()
vae.compile(optimizer='adam', loss='categorical_crossentropy')

vae.fit(all_numerical_positions,
        all_numerical_positions,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1)

vae.save(f'models/middle_{MIDDLE_DIMENSIONS}/vae_all.h5')
encoder.save(f'models/middle_{MIDDLE_DIMENSIONS}/vae_only_encoder.h5')
decoder.save(f'models/middle_{MIDDLE_DIMENSIONS}/vae_only_decoder.h5')
