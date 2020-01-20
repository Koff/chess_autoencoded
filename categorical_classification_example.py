import numpy as np

from keras.layers import Input, Dense, Conv2D, UpSampling2D, AveragePooling2D
from keras.utils import to_categorical
from keras import backend as K, Sequential, Model
from keras.losses import mse, binary_crossentropy, categorical_crossentropy
from keras import optimizers
import matplotlib.pyplot as plt


# Generate dummy data
data_1 = np.random.randint(2, size=(10000, 256)).reshape((10000, 8, 8, 4)).astype(np.int8)

train, validate = data_1[:8000], data_1[8000:]

# this is our input placeholder
input_pos = Input(shape=(8, 8, 4,))

# "encoded" is the encoded representation of the input
encoded = Conv2D(64*4, (3, 3), activation='linear', padding='same')(input_pos)
encoded = AveragePooling2D((2, 2), padding='same')(encoded)
encoded = Conv2D(32*4, (3, 3), activation='linear', padding='same')(encoded)
encoded = AveragePooling2D((2, 2), padding='same')(encoded)
encoded = Conv2D(16*4, (3, 3), activation='linear', padding='same')(encoded)
encoded = AveragePooling2D((2, 2), padding='same')(encoded)

# "decoded" is the lossy reconstruction of the input
decoded = Conv2D(16*4, (3, 3), activation='linear', padding='same')(encoded)
decoded = UpSampling2D((2, 2))(decoded)
decoded = Conv2D(32*4, (3, 3), activation='linear', padding='same')(decoded)
decoded = UpSampling2D((2, 2))(decoded)
decoded = Conv2D(64*4, (3, 3), activation='linear', padding='same')(decoded)
decoded = UpSampling2D((2, 2))(decoded)
decoded = Conv2D(4, (3, 3), activation='linear', padding='same')(decoded)


autoencoder = Model(input_pos, decoded)
autoencoder.summary()

opt = optimizers.Adam()

autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(train, train,
                epochs=5,
                batch_size=50,
                )
pass

fig = plt.figure(figsize=(8, 4))
fig.add_subplot(2, 1, 1)
plt.imshow(train[100].reshape((8, 32)), cmap='gray')
fig.add_subplot(2, 1, 2)
predicted = autoencoder.predict(train[100].reshape((1, 8, 8, 4))).reshape((8, 32))
predictions_labels = np.round(predicted[:, :]).astype(int)
plt.imshow(predictions_labels, cmap='gray')
plt.show()
