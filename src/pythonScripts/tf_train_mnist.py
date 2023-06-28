import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


def preprocess_data(x, y):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 28 * 28)
    x = x.astype("float32") / 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = np_utils.to_categorical(y)
    y = y.reshape(y.shape[0], 10)
    return x, y


model = tf.keras.Sequential(
    [
        layers.Input((784,)),
        layers.Dense(40, activation="tanh", name="layer0", use_bias=False),
        layers.Dense(10, activation="tanh", name="layer3", use_bias=False),
    ]
)

model.compile("SGD", "mse")

x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)
model.summary()

hist = model.fit(x_train, y_train, epochs=5, batch_size=16)
print(model.evaluate(x_test, y_test))

np.zeros((2, 2))

f = open("std_weights.txt", "w", encoding="ascii")
f.write("nn;784;40;10\n")
L = []
for l in model.layers[1:]:
    tmp = []
    w = (l.get_weights())
    w = np.array(w).flat()
    print(w.shape)
    a = np.zeros((w.shape[0] + 1, w.shape[0]))
    for i in range(w.shape[0]):
        # tmp.append(w[i])
        for j, wt in enumerate(w):
            if (j != w.shape[1] - 1 and i != w.shape[0] - 1):
                f.write(str(wt) + ";")
            else:
                f.write(str(wt))
        f.write("\n")
