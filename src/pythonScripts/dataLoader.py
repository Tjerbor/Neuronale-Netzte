import numpy as np
import tensorflow as tf
from keras.utils import np_utils


# f = open()


# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


def writeData(train_filepath, x_train, y_train):
    f_train = open(train_filepath, "w", encoding="ascii")
    count = 0
    for x_sample, y_sample in zip(x_train, y_train):
        for i, xs in enumerate(x_sample):
            if (i != len(x_sample) - 1):
                f_train.write(str(xs) + ";")
            else:
                f_train.write(str(xs))
        f_train.write("\t")
        for ys in y_sample:
            if (i != len(y_sample) - 1):
                f_train.write(str(ys) + ";")
            else:
                f_train.write(str(ys))

        count += 1
        if (count != len(x_train)):
            f_train.write("\n")

    print("wrote File: " + train_filepath)


def preprocess_data(x, y):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 28 * 28)
    x = x.astype("float32") / 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = np_utils.to_categorical(y)
    y = y.reshape(y.shape[0], 10)
    return x, y


def writeColour():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    X = np.concatenate([x_train, x_test])
    X = X / 127.5 - 1

    # Set reshaped array to X
    X = X.reshape((70000, 28, 28, 1))

    # Convert images and store them in X3

    f_train_fpath = "./mnist_colour2.txt"
    f_train = open(f_train_fpath, "w", encoding="ascii")

    y = np.concatenate([y_train, y_test])
    y = np_utils.to_categorical(y)
    y = y.reshape(y.shape[0], 10)

    del (x_train, y_train, x_test, y_test)

    maxC = 70000
    count = 0
    X = np.array(X)

    for x_sample, y_sample in zip(X, y):
        x_sample = tf.cast(x_sample, dtype=tf.float32)
        x_sample = tf.image.grayscale_to_rgb(
            x_sample,
            name=None
        )
        x_sample = np.array(x_sample)

        for i, xs in enumerate(x_sample):
            if (i != len(x_sample) - 1):
                f_train.write(str(xs) + ";")
            else:
                f_train.write(str(xs))
        f_train.write("|")
        for ys in y_sample:
            if (i != len(y_sample) - 1):
                f_train.write(str(ys) + ";")
            else:
                f_train.write(str(ys))

        count += 1
        f_train.write("\n")
        if (count == maxC):
            break


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)

train_filepath = "../train_mnist.txt"
test_filepath = "../test_mnist.txt"

writeData(train_filepath, x_train, y_train)
writeData(test_filepath, x_test, y_test)

# writeColour()
