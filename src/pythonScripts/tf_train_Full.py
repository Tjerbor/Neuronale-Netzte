import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# https://stackoverflow.com/questions/2448015/2d-convolution-using-python-and-numpy

def convolution2d(image, kernel, bias):
    m, n = kernel.shape
    if (m == n):
        y, x = image.shape
        y = y - m + 1
        x = x - m + 1
        new_image = np.zeros((y, x))
        for i in range(y):
            for j in range(x):
                new_image[i][j] = np.sum(image[i:i + m, j:j + m] * kernel) + bias
    return new_image


def preprocess_data(x, y):
    # reshape and normalize input data
    x = x.astype("float32") / 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = np_utils.to_categorical(y)
    y = y.reshape(y.shape[0], 10)
    return x, y


model = tf.keras.Sequential(
    [
        layers.Input((28, 28,)),
        layers.Conv1D(8, kernel_size=3, activation="tanh", name="conv"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(4),
        layers.Flatten(),
        layers.Dense(10, activation="tanh")
    ]
)

model.compile("SGD", "mse")

x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)
model.summary()

hist = model.fit(x_train, y_train, epochs=5, batch_size=16)
print(model.evaluate(x_test, y_test))

np.zeros((2, 2))
