import tensorflow as tf
from keras.utils import np_utils

train_filepath = "../train_mnist.txt"
test_filepath = "../test_mnist.txt"

# f = open()


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


def writeData(train_filepath, x_train, y_train):
    f_train = open(train_filepath, "w", encoding="ascii")
    count = 0
    for x_sample, y_sample in zip(x_train, y_train):
        for i, xs in enumerate(x_sample):
            if (i != len(x_sample) - 1):
                f_train.write(str(xs) + ",")
            else:
                f_train.write(str(xs))
        f_train.write("|")
        for ys in y_sample:
            if (i != len(y_sample) - 1):
                f_train.write(str(ys) + ",")
            else:
                f_train.write(str(ys))

        count += 1
        if (count != len(x_train)):
            f_train.write("\n")


def preprocess_data(x, y):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 28 * 28)
    x = x.astype("float32") / 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = np_utils.to_categorical(y)
    y = y.reshape(y.shape[0], 10)
    return x, y


x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)

writeData(train_filepath, x_train, y_train)
writeData(test_filepath, x_test, y_test)
