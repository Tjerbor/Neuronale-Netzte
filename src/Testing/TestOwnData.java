package Testing;

import Train.LoadOwn;
import function.TanH;
import layer.FullyConnectedLayer;
import load.LoadModel;
import main.NeuralNetwork;
import utils.Reader;

import java.awt.image.BufferedImage;
import java.io.IOException;

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

        for (FullyConnectedLayer layer : layers) {
            layer.setActivation(new TanH());
        }

        double[][] test = new double[x_test.length][784];

        for (int i = 0; i < x_test.length; i++) {
            for (int j = 0; j < x_test[0].length; j++) {
                test[i][j] = x_test[i][783 - j];
            }
        }

        NeuralNetwork nn = new NeuralNetwork();

        NeuralNetwork neuralNetwork = LoadModel.loadModel("nn_emnist_digits_weights.txt");

        nn.create(layers);

        //nn.test(test, y_test);
        neuralNetwork.test(x_test, y_test);
        neuralNetwork.test(test, y_test);


    }

    public static void main(String[] args) throws IOException {

        main2();


    }
}
