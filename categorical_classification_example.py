import numpy as np

from keras.layers import Input, Conv2D, UpSampling2D, AveragePooling2D
from keras import Model
from keras import optimizers
import matplotlib.pyplot as plt


# Load data
all_numerical_positions = np.load('x_data.npy')

# Cut data to cut training time
all_numerical_positions = all_numerical_positions[:int(np.floor(all_numerical_positions.shape[0]/10)), :, :, :]

# Transform integers to 4-bit binary
m = 4
all_numerical_positions = (((all_numerical_positions[:, None] & (1 << np.arange(m)))) > 0).astype(int)
all_numerical_positions = all_numerical_positions.reshape((-1, 8, 8, 4))

# this is our input placeholder
input_pos = Input(shape=(8, 8, 4,))

# "encoded" is the encoded representation of the input
encoded = Conv2D(64 * 4, (3, 3), activation='linear', padding='same')(input_pos)
encoded = AveragePooling2D((2, 2), padding='same')(encoded)
encoded = Conv2D(32 * 4, (3, 3), activation='linear', padding='same')(encoded)
encoded = AveragePooling2D((2, 2), padding='same')(encoded)
encoded = Conv2D(16 * 4, (3, 3), activation='linear', padding='same')(encoded)
encoded = AveragePooling2D((2, 2), padding='same')(encoded)

# "decoded" is the lossy reconstruction of the input
decoded = Conv2D(16 * 4, (3, 3), activation='linear', padding='same')(encoded)
decoded = UpSampling2D((2, 2))(decoded)
decoded = Conv2D(32 * 4, (3, 3), activation='linear', padding='same')(decoded)
decoded = UpSampling2D((2, 2))(decoded)
decoded = Conv2D(64 * 4, (3, 3), activation='linear', padding='same')(decoded)
decoded = UpSampling2D((2, 2))(decoded)
decoded = Conv2D(4, (3, 3), activation='linear', padding='same')(decoded)


autoencoder = Model(input_pos, decoded)
autoencoder.summary()

opt = optimizers.Adam()

autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(all_numerical_positions,
                all_numerical_positions,
                epochs=2,
                batch_size=256,
                )
pass

fig = plt.figure(figsize=(8, 4))
fig.add_subplot(2, 1, 1)
plt.imshow(all_numerical_positions[100].reshape((8, 32)), cmap='gray')
fig.add_subplot(2, 1, 2)
predicted = autoencoder.predict(all_numerical_positions[100].reshape((1, 8, 8, 4))).reshape((8, 32))
predictions_labels = np.round(predicted[:, :]).astype(int)
plt.imshow(predictions_labels, cmap='gray')
plt.show()
