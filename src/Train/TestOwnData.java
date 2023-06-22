package Train;

import layer.FullyConnectedLayer;
import layer.TanH;
import utils.Reader;
import utils.Utils;

import java.awt.image.BufferedImage;
import java.io.IOException;

import static Train.LoadOwn.getTestData;
import static Train.LoadOwn.loadSingle;

public class TestOwnData {


    public static double[][] convertRGBImageToGrayscaleArray(BufferedImage image) {
        double[][] grayscale = new double[image.getWidth()][image.getHeight()];

        for (int x = 0; x < image.getWidth(); x++) {
            for (int y = 0; y < image.getHeight(); y++) {
                int rgb = image.getRGB(x, y);
                grayscale[y][x] = 0.299 * (double) ((rgb & 0x00ff0000) >> 16) + //Red
                        0.587 * (double) ((rgb & 0x0000ff00) >> 8) + //Green
                        0.114 * (double) (rgb & 0x000000ff); //Blue
            }
        }
        return grayscale;
    }


    public static void main2() throws IOException {

        double[][][] testData;
        String dirFpath = "data/selfmade_data";
        testData = LoadOwn.getImages(dirFpath);

        double[][] y_test = testData[1];
        double[][] x_test = testData[0];

        String weights = "weights_0.95266664_.txt";
        FullyConnectedLayer[] layers = Reader.create(weights);

        for (int i = 0; i < layers.length; i++) {
            layers[i].setActivation(new TanH());
        }


        double[][] test = new double[x_test.length][784];

        for (int i = 0; i < x_test.length; i++) {
            for (int j = 0; j < x_test[0].length; j++) {
                test[i][j] = x_test[i][783 - j];
            }
        }


        double[] out;
        double loss_per_step = 0;
        for (int ti = 0; ti < x_test.length; ti++) {


            out = test[ti];

            for (FullyConnectedLayer l : layers) {

            }
            for (int i = 0; i < layers.length; i++) {
                out = layers[i].forward(out);
            }

            System.out.println("Predicted Class: " + Utils.argmax(out));
            System.out.println("Actual Class: " + Utils.argmax(y_test[ti]));
            System.out.println("");

            if (Utils.argmax(out) == Utils.argmax(y_test[ti])) {
                loss_per_step += 1;

            }
        }
        System.out.println("Acc: " + loss_per_step / x_test.length);

        double[][] data = loadSingle("src/pythonScripts/test_0.txt");

        out = data[0];
        for (int i = 0; i < layers.length; i++) {
            out = layers[i].forward(out);
        }

        System.out.println("Predicted Class: " + Utils.argmax(out));
        System.out.println("Actual Class: " + Utils.argmax(data[1]));
        System.out.println("");

        if (Utils.argmax(out) == Utils.argmax(data[1])) {
            loss_per_step += 1;

        }

    }

    public static void main(String[] args) throws IOException {

        main2();


    }

    public static void main3(String[] args) throws IOException {


        String weights = "weights_0.9605333_.txt";
        FullyConnectedLayer[] layers = Reader.create(weights);
        String dirFpath = "src/Train/OwnData";
        double[][][] testData = getTestData(dirFpath);


        for (int i = 0; i < layers.length; i++) {
            layers[i].setActivation(new TanH());
        }

        double[][] y_test = testData[1];
        double[][] x_test = testData[0];


        double[][] test = new double[x_test.length][784];

        for (int i = 0; i < x_test.length; i++) {
            for (int j = 0; j < x_test[0].length; j++) {
                test[i][j] = x_test[i][783 - j];
            }
        }


        double[] out;
        double loss_per_step = 0;
        for (int ti = 0; ti < x_test.length; ti++) {


            out = test[ti];

            for (FullyConnectedLayer l : layers) {

            }
            for (int i = 0; i < layers.length; i++) {
                out = layers[i].forward(out);
            }

            System.out.println("Predicted Class: " + Utils.argmax(out));
            System.out.println("Actual Class: " + Utils.argmax(y_test[ti]));
            System.out.println("");

            if (Utils.argmax(out) == Utils.argmax(y_test[ti])) {
                loss_per_step += 1;

            }
        }
        System.out.println("Acc: " + loss_per_step / x_test.length);

        double[][] data = loadSingle("src/pythonScripts/test_0.txt");

        out = data[0];
        for (int i = 0; i < layers.length; i++) {
            out = layers[i].forward(out);
        }

        System.out.println("Predicted Class: " + Utils.argmax(out));
        System.out.println("Actual Class: " + Utils.argmax(data[1]));
        System.out.println("");

        if (Utils.argmax(out) == Utils.argmax(data[1])) {
            loss_per_step += 1;

        }

    }
}
