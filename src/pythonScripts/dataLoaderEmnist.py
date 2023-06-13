from keras.utils import np_utils


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

    print("Wrote File: " + train_filepath)


def preprocess_data(x, y):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 28 * 28)
    x = x.astype("float32") / 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = np_utils.to_categorical(y)
    classes = len(y[0])
    y = y.reshape(y.shape[0], classes)
    return x, y


from emnist import extract_training_samples, extract_test_samples, list_datasets

type_ = "balanced"

print(list_datasets())

train_filepath = f"../train_emnist_{type_}.txt"
test_filepath = f"../test_emnist_{type_}.txt"

x_train, y_train = extract_training_samples(type_)
x_train, y_train = preprocess_data(x_train, y_train)
writeData(train_filepath, x_train, y_train)

x_test, y_test = extract_test_samples(type_)
x_test, y_test = preprocess_data(x_test, y_test)
writeData(test_filepath, x_test, y_test)
