import tensorflow as tf
from keras.utils import np_utils
from matplotlib import pyplot as plt
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


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
        layers.Input((28, 28)),
        layers.Conv1D(32, kernel_size=3, activation="relu", name="conv"),
        layers.MaxPooling1D(4),
        layers.Flatten(),
        layers.Dense(10, activation="softmax", name="out"),
    ]
)

# optimizer Options: SGD, Adam, rmsprop, AdaGard,
# loss: mse, CategoricalCrossentropy
model.compile("SGD", "CategoricalCrossentropy",
              metrics=['accuracy'])

x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)
model.summary()

hist = model.fit(x_train, y_train, epochs=10, batch_size=16)
# model.predict(y_test, y_train)

plt.plot(hist.history['accuracy'])
# plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
