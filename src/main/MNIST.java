package main;

import java.io.*;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.zip.GZIPInputStream;

/**
 * This class contains utility methods for reading the <a href="http://yann.lecun.com/exdb/mnist/">MNIST database of handwritten digits</a>.
 */
public final class MNIST {
    private MNIST() {
    }

    private static final Logger LOGGER = Logger.getLogger(MNIST.class.getName());

    /**
     * Each image in the MNIST database is 28 by 28 pixels, for a total of 784 pixels.
     */
    private static final int PIXELS = 784;
    /**
     * The MNIST database classifies the digits 0 to 9.
     */
    private static final int DIGITS = 10;

    /**
     * This method converts a byte array into an integer.
     */
    private static int bytesToInt(byte[] number) {
        if (number.length > 4) {
            throw new IllegalArgumentException("The length of the array exceeds the size of an integer.");
        }

        int result = 0;

        for (byte b : number) {
            result = (result << 8) + (b & 0xFF);
        }

        return result;
    }

    /**
     * This method attempts to read the given MNIST data sets.
     * It throws an exception if the file does not exist, an I/O error occurs, or the format is not correct.
     * The return value is a three-dimensional array of length two containing the image and label data.
     *
     * @param idx3 Image Data
     * @param idx1 Label Data
     * @see <a href="http://yann.lecun.com/exdb/mnist/">MNIST database of handwritten digits</a>
     */
    public static double[][][] read(String idx3, String idx1) throws IOException {
        try (
                InputStream imagesIn = new GZIPInputStream(new FileInputStream(idx3));
                InputStream labelsIn = new GZIPInputStream(new FileInputStream(idx1))
        ) {

            byte[] imagesMeta = new byte[16];
            byte[] labelsMeta = new byte[8];

            if (imagesIn.read(imagesMeta, 0, 16) != 16) {
                throw new IllegalArgumentException("The metadata of the file " + idx3 + " does not comply with the MNIST idx3 format.");
            }

            if (labelsIn.read(labelsMeta, 0, 8) != 8) {
                throw new IllegalArgumentException("The metadata of the file " + idx1 + " does not comply with the MNIST idx1 format.");
            }

            int length = bytesToInt(Arrays.copyOfRange(imagesMeta, 4, 8));

            if (length != bytesToInt(Arrays.copyOfRange(labelsMeta, 4, 8))) {
                throw new IllegalArgumentException("The number of items in the two files does not match.");
            }

            byte[] buffer = new byte[1];

            double[][] images = new double[length][PIXELS];

            double[][] labels = new double[length][DIGITS];

            for (int i = 0; i < length; i++) {
                for (int j = 0; j < PIXELS; j++) {
                    if (imagesIn.read(buffer, 0, 1) != 1) {
                        throw new IllegalArgumentException("The length of the file " + idx3 + " does not match the metadata.");
                    }

                    images[i][j] = bytesToInt(buffer) / 255.;
                }

                if (labelsIn.read(buffer, 0, 1) != 1) {
                    throw new IllegalArgumentException("The length of the file " + idx1 + " does not match the metadata.");
                }

                labels[i][bytesToInt(buffer)] = 1;
            }

            LOGGER.log(Level.INFO, "{0} Images & Labels were read successfully.", length);

            return new double[][][]{images, labels};
        }
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
