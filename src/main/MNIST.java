package main;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.zip.GZIPInputStream;

/**
 * This class contains utility methods for reading the <a href="http://yann.lecun.com/exdb/mnist/">MNIST database of handwritten digits</a>.
 * It can also be used to read the <a href="https://www.nist.gov/itl/products-and-services/emnist-dataset">EMNIST database</a>.
 */
public final class MNIST {
    private static final Logger LOGGER = Logger.getLogger(MNIST.class.getName());
    /**
     * Each image in the MNIST database is 28 by 28 pixels, for a total of 784 pixels.
     */
    private static final int PIXELS = 784;
    /**
     * The MNIST database classifies the digits 0 to 9.
     */
    private static final int DIGITS = 10;

    private MNIST() {
    }

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
     * This method attempts to read the given EMNIST data sets.
     * The number of classes must be passed as an additional parameter.
     *
     * <ul>
     *     <li>ByClass: 814,255 Characters, 62 Unbalanced Classes</li>
     *     <li>ByMerge: 814,255 Characters, 47 Unbalanced Classes</li>
     *     <li>Balanced: 131,600 Characters, 47 Balanced Classes</li>
     *     <li>Letters: 145,600 Characters, 26 Balanced Classes</li>
     *     <li>Digits: 280,000 Characters, 10 Balanced Classes</li>
     *     <li>MNIST: 70,000 Characters, 10 Balanced Classes</li>
     * </ul>
     *
     * @see <a href="https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/Readme.txt">README</a>
     * @see <a href="https://arxiv.org/pdf/1702.05373v1.pdf">EMNIST Paper</a>
     * @see MNIST#read(String, String)
     */
    public static double[][][] read(String idx3, String idx1, int number) throws IOException {
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

            double[][] labels = new double[length][number];

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

    /**
     * This method attempts to read the given MNIST data sets.
     * It throws an exception if the file does not exist, an I/O error occurs, or the format is not correct.
     * The return value is a three-dimensional array of length two containing the image and label data.
     *
     * @param idx3 Image Data
     * @param idx1 Label Data
     * @see MNIST#read(String, String, int)
     */
    public static double[][][] read(String idx3, String idx1) throws IOException {
        return read(idx3, idx1, DIGITS);
    }
}
