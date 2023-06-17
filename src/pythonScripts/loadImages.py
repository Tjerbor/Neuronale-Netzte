import cv2
import numpy as np
import tensorflow as tf


def writeSingleImage(x_sample):
    s = ""
    for i, xs in enumerate(x_sample):
        if (i != len(x_sample) - 1):
            s += (str(xs) + ";")
        else:
            s += (str(xs))

    return s


np.zeros(1)

image = cv2.imread("/home/dblade/Documents/Neuronale-Netzte/src/Train/Images/sample02_0.png", 0)  # read grayscale

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(image.shape)

image = image.astype(np.float32)

flatt = image.flatten()

image /= 255

a = np.ones((28, 28))

a = a - image

flatt = flatt.tolist()

flatt = np.array(flatt)

flatt1 = np.ones(784)

cv2.imshow('unchanged image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()

print(len(flatt))

print(np.max(flatt1))
print(np.min(flatt1))

print(flatt1)

y = [0] * 10
y[0] = 1
s = writeSingleImage(flatt1)
f_train = open("test_0.txt", "w", encoding="ascii")
f_train.write(s)


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
