import numpy as np
import keras

from keras import Model
from keras.layers import Input, Conv2D, UpSampling2D, AveragePooling2D, Dense


EPOCHS = 4
BATCH_SIZE = 128

# Load chess positions data
all_numerical_positions = np.load('x_data.npy')

# Remove data to cut training time
# all_numerical_positions = all_numerical_positions[:int(np.floor(all_numerical_positions.shape[0] / 20)), :, :, :]

# Transform integers to 4-bit binary
m = 4
all_numerical_positions = (((all_numerical_positions[:, None] & (1 << np.arange(m)))) > 0).astype(int)
all_numerical_positions = all_numerical_positions.reshape((-1, 8, 8, 4))


# Input into encoder
input_pos = Input(shape=(8, 8, 4,))

# "encoded" is the encoded representation of the input
encoded = Conv2D(64 * 4, (3, 3), activation='linear', padding='same')(input_pos)
encoded = AveragePooling2D((2, 2), padding='same')(encoded)
encoded = Conv2D(32 * 4, (3, 3), activation='linear', padding='same')(encoded)
encoded = AveragePooling2D((2, 2), padding='same')(encoded)
encoded = Conv2D(16 * 4, (3, 3), activation='linear', padding='same')(encoded)
encoded = AveragePooling2D((2, 2), padding='same')(encoded)

# Bottle neck
encoded = Dense(1, input_shape=(None, 1, 1, 64))(encoded)

# Decoder input
decoded_input = Input(shape=(1, 1, 1,))

# Rest of decoder
decoded = Conv2D(16 * 4, (3, 3), activation='linear', padding='same')(decoded_input)
decoded = UpSampling2D((2, 2))(decoded)
decoded = Conv2D(32 * 4, (3, 3), activation='linear', padding='same')(decoded)
decoded = UpSampling2D((2, 2))(decoded)
decoded = Conv2D(64 * 4, (3, 3), activation='linear', padding='same')(decoded)
decoded = UpSampling2D((2, 2))(decoded)
decoded = Conv2D(4, (3, 3), activation='linear', padding='same')(decoded)

encoder = Model(input_pos, encoded, name='encoder')
encoder.summary()

decoder = Model(decoded_input, decoded, name='decoder')
decoder.summary()

# Variational autoencoder definition
outputs = decoder(encoder(input_pos))
vae = keras.Model(input_pos, outputs, name='vae_mlp')

vae.summary()
vae.compile(optimizer='adam', loss='mse')

vae.fit(all_numerical_positions, all_numerical_positions,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE)

vae.save('models/vae_2_dimensions_encoded.h5')
encoder.save('models/vae_only_encoder.h5')
decoder.save('models/vae_only_decoder.h5')
