package main;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

/**
 * This class contains utility methods for reading the <a href="http://yann.lecun.com/exdb/mnist/">MNIST database of handwritten digits</a>.
 */
public final class EMNIST {
    /**
     * Each image in the MNIST database is 28 by 28 pixels, for a total of 784 pixels.
     */
    private static final int PIXELS = 784;
    /**
     * The MNIST database classifies the digits 0 to 9.
     */
    private static final int DIGITS = 47;

    private EMNIST() {
    }

    /**
     * This method attempts to read the MNIST data set with the given size.
     * The MNIST training set contains 60,000 examples and the MNIST test set contains 10,000 examples.
     * The method throws an exception if the file does not exist, an I/O error occurs, or the format is not correct.
     * <p>
     * The return value is a three-dimensional array of length two.
     * The two two-dimensional arrays in turn contain the input data and the corresponding classifications.
     */
    public static double[][][] read(String path, int size) throws IOException {
        double[][] pixels = new double[size][PIXELS];
        double[][] digits = new double[size][DIGITS];

        int i = 0;

        try (BufferedReader in = new BufferedReader(new FileReader(path))) {
            String line;

            while ((line = in.readLine()) != null) {
                String[] data = line.split("\t");

                String[] x = data[0].split(";");
                String[] y = data[1].split(";");

                if (x.length != PIXELS || y.length != DIGITS) {
                    throw new IllegalArgumentException("The file " + path + " does not conform to the MNIST format.");
                }

                pixels[i] = new double[PIXELS];
                digits[i] = new double[DIGITS];

                for (int j = 0; j < x.length; j++) {
                    pixels[i][j] = Double.parseDouble(x[j]);
                }

                for (int j = 0; j < y.length; j++) {
                    digits[i][j] = Double.parseDouble(y[j]);
                }

                if (++i == size) {
                    break;
                }
            }
        }

        return new double[][][]{pixels, digits};
    }

    public static double[][][] x_train_2_batch(double[][] x_train, int batch_size) throws Exception {

        int data_size = x_train.length;

        if (data_size % batch_size != 0) {
            throw new IllegalArgumentException("Batch size must be equally dividable");
        }


        int firstShape = data_size / batch_size;
        double[][][] x_train_out = new double[firstShape][batch_size][x_train[0].length];
        int count = 0;
        double[][] tmp = new double[batch_size][x_train[0].length];
        for (int i = 0; i < data_size; i += batch_size) {
            for (int j = 0; j < batch_size; j++) {
                x_train_out[count][j] = x_train[i + j];

            }

            count += 1;
        }


        return x_train_out;
    }

    public static double[][][] y_train_2_batch(double[][] y_train, int batch_size) throws Exception {

        int data_size = y_train.length;

        if (data_size % batch_size != 0) {
            throw new IllegalArgumentException("Batch size musst be equally dividable");
        }


        int firstShape = data_size / batch_size;
        double[][][] y_train_out = new double[firstShape][batch_size][y_train[0].length];
        int count = 0;
        double[][] tmp = new double[batch_size][y_train[0].length];
        for (int i = 0; i < data_size; i += batch_size) {
            for (int j = 0; j < batch_size; j++) {
                y_train_out[count][j] = y_train[i + j];

            }

            count += 1;
        }


        return y_train_out;
    }
}
