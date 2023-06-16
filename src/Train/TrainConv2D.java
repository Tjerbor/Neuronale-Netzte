package Train;

import layer.*;
import main.MNIST;
import utils.Array_utils;
import utils.Utils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class TrainConv2D {

    public static void export(String filename, Conv2D c, FullyConnectedLayer f1) {

        StringBuilder s = new StringBuilder();
        s.append("Topology: ").append("layers;").append("\n");


        s.append(f1.toString(true));
        s.append("\n");
        s.append("\n");
        s.append(c.toString());


        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            writer.write(s.toString());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }

    public static void main(String[] args) throws Exception {


        String fpath = "/home/dblade/Documents/Neuronale-Netzte/src/train_mnist.txt";
        double[][][] trainingData = MNIST.read(fpath, 60000);
        double[][] x_train = trainingData[0];
        double[][] y_train = trainingData[1];


        double[][][][] x_train_conv = genConvData(x_train);


        String fpath_test = "/home/dblade/Documents/Neuronale-Netzte/src/test_mnist.txt";
        double[][][] testData = MNIST.read(fpath_test, 15000);
        double[][] x_test = testData[0];
        double[][] y_test = testData[1];

        double[][][][] x_test_conv = genConvData(x_test);

        int convFilter = 8;
        int outSize = 12;
        Conv2D conv2D = new Conv2D(convFilter, new int[]{1, 28, 28}, 5);
        //Conv2D conv2D_2 = new Conv2D(convFilter, conv2D.getOutputShape(), 5);
        Conv2dActivation convAct = new Conv2dActivation(new ReLu());
        //Conv2dActivation convAct2 = new Conv2dActivation(new TanH());
        conv2D.setUseBiases(false);
        //conv2D.setActivation(new ReLu());
        MaxPool2D pool = new MaxPool2D(conv2D.getOutputShape(), 2);

        int size = Array_utils.sumUpMult(pool.getOutputShape());

        FullyConnectedLayer f1 = new FullyConnectedLayer(size, 10, "tanh");

        f1.setActivation(new TanH());

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

            //learning_rate -= (learning_rate * 0.05);
            for (int j = 0; j < step_size; j++) {

                out = Array_utils.copyArray(x_train_conv[j]);
                out = conv2D.forward(out);
                out = convAct.forward(out);

                //out = conv2D_2.forward(out);
                //out = convAct2.forward(out);
                out = pool.forward(out);
                flatOut = Array_utils.flatten(out);

                flatOut = f1.forward(flatOut);

                loss_per_step += loss.forward(flatOut, y_train[j]);

                flatOut = loss.backward(flatOut, y_train[j]);

                flatOut = f1.backward(flatOut, learning_rate);


                out = Array_utils.reFlat(flatOut, new int[]{convFilter, outSize, outSize});


                out = pool.backPropagation(out);
                //out = convAct2.backward(out);
                //out = conv2D_2.backward(out, learning_rate);

                out = convAct.backward(out);
                conv2D.backward(out, learning_rate);
            }
            System.out.println("Loss: " + loss_per_step / x_train.length);
            System.out.println("Time: " + ((System.currentTimeMillis() - st) / 1000));

            double[][][] outT;
            double[] flatOutT;
            double loss_per_stepT = 0;
            for (int ti = 0; ti < x_test.length; ti++) {

                outT = Array_utils.copyArray(x_test_conv[ti]);
                outT = conv2D.forward(outT);
                //outT = pool.forward(outT);
                flatOutT = Array_utils.flatten(outT);

                flatOutT = f1.forward(flatOutT);


                if (Utils.argmax(flatOutT) == Utils.argmax(y_test[ti])) {
                    loss_per_stepT += 1;

                }
            }
            System.out.println("Acc: " + loss_per_stepT / x_test.length);

        }

        double[][][] out;
        double[] flatOut;
        double loss_per_step = 0;
        for (int ti = 0; ti < x_test.length; ti++) {

            out = Array_utils.copyArray(x_test_conv[ti]);
            out = conv2D.forward(out);
            //out = pool.forward(out);
            flatOut = Array_utils.flatten(out);

            flatOut = f1.forward(flatOut);


            if (Utils.argmax(flatOut) == Utils.argmax(y_test[ti])) {
                loss_per_step += 1;

            }
        }
        System.out.println("Acc: " + loss_per_step / x_test.length);
        String outFPath = "conv_weights_" + loss_per_step / x_test.length + ".txt";
        export(outFPath, conv2D, f1);

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


}
