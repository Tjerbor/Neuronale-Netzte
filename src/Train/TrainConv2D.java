package Train;

import layer.Conv2D;
import layer.FullyConnectedLayer;
import layer.MSE;
import layer.MaxPooling2DNew;
import main.MNIST;
import utils.Array_utils;
import utils.Utils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class TrainConv2D {

    public static void export(String filename, FullyConnectedLayer[] layers) {

        StringBuilder s = new StringBuilder();
        s.append("Topology: ").append("layers;").append("\n");
        for (FullyConnectedLayer layer : layers) {

            s.append(layer.toString(true));
            s.append("\n");


        }

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            writer.write(s.toString());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }

    public static void main(String[] args) throws Exception {


        String fpath = "./src/TrainData/train_mnist.txt";
        double[][][] trainingData = MNIST.read(fpath, 60000);
        double[][] x_train = trainingData[0];
        double[][] y_train = trainingData[1];


        double[][][][] x_train_conv = genConvData(x_train);


        String fpath_test = "./src/TrainData/test_mnist.txt";
        double[][][] testData = MNIST.read(fpath_test, 15000);
        double[][] x_test = testData[0];
        double[][] y_test = testData[1];

        double[][][][] x_test_conv = genConvData(x_test);

        int convFilter = 8;

        Conv2D conv2D = new Conv2D(convFilter, new int[]{1, 28, 28});
        conv2D.setUseBiases(false);
        conv2D.setActivation("relu");
        MaxPooling2DNew pool = new MaxPooling2DNew();
        FullyConnectedLayer f1 = new FullyConnectedLayer(13 * 13 * convFilter, 40);
        //FullyConnectedLayer f2 = new FullyConnectedLayer(80, 40, "tanh");
        FullyConnectedLayer f3 = new FullyConnectedLayer(40, 10, "tanh");

        MSE loss = new MSE();
        double learning_rate = 0.1;
        int epochs = 5;
        int step_size = x_train_conv.length;


        long st;

        System.out.println("Started Training");

        for (int i = 0; i < epochs; i++) {
            double[][][] out;
            double[] flatOut;
            double loss_per_step = 0;
            st = System.currentTimeMillis();
            for (int j = 0; j < step_size; j++) {

                out = Array_utils.copyArray(x_train_conv[j]);
                out = conv2D.forward(out);
                out = pool.forward(out);
                flatOut = Array_utils.flatten(out);

                flatOut = f1.forward(flatOut);
                //flatOut = f2.forward(flatOut);
                flatOut = f3.forward(flatOut);


                loss_per_step += loss.forward(flatOut, y_train[j]);

                flatOut = loss.backward(flatOut, y_train[j]);

                flatOut = f3.backward(flatOut, learning_rate);
                //flatOut = f2.backward(flatOut, learning_rate);
                flatOut = f1.backward(flatOut, learning_rate);


                out = reFlat(flatOut, new int[]{convFilter, 13, 13});
                out = pool.backward(out);
                conv2D.backward(out, learning_rate);
            }
            System.out.println("Loss: " + loss_per_step / x_train.length);
            System.out.println("Time: " + ((System.currentTimeMillis() - st) / 1000));

        }

        double[][][] out;
        double[] flatOut;
        double loss_per_step = 0;
        for (int ti = 0; ti < x_test.length; ti++) {

            out = Array_utils.copyArray(x_test_conv[ti]);
            out = conv2D.forward(out);
            out = pool.forward(out);
            flatOut = Array_utils.flatten(out);

            flatOut = f1.forward(flatOut);
            //flatOut = f2.forward(flatOut);
            flatOut = f3.forward(flatOut);

            if (Utils.argmax(flatOut) == Utils.argmax(y_test[ti])) {
                loss_per_step += 1;

            }
        }
        System.out.println("Acc: " + loss_per_step / x_test.length);


        /**
         * FullyConnectedLayer[] layers = new FullyConnectedLayer[]{f1, f2};
         *         String outFPath = "weights_" + loss_per_step / x_test.length + ".txt";
         *         export(outFPath, layers);
         */


    }

    public static double[][][][] genConvData(double[][] x_train) {

        double[][][][] out = new double[x_train.length][1][28][28];

        int count = 0;
        for (int i = 0; i < x_train.length; i++) {
            count = 0;
            for (int j = 0; j < 28; j++) {
                for (int k = 0; k < 28; k++) {
                    out[i][0][j][k] = x_train[i][count];
                    count += 1;
                }
            }
        }

        return out;
    }

    public static double[][][] reFlat(double[] a, int[] shape) {

        double[][][] c = new double[shape[0]][shape[1]][shape[2]];
        int count = 0;
        for (int fi = 0; fi < shape[0]; fi++) {
            for (int i = 0; i < shape[1]; i++) {
                for (int j = 0; j < shape[2]; j++) {
                    c[0][i][j] = a[count];
                    count += 1;
                }
            }
        }


        return c;
    }
}
