package Train;

import layer.*;
import main.MNIST;
import main.NeuralNetwork;
import utils.Utils;

public class TrainMnistConv {


    public static double[][] reshape(double[] x) {

        int d = (int) Math.sqrt(x.length);
        double[][] c = new double[d][d];


        int count = 0;
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                c[i][j] = x[count];
                count += 1;

            }
        }

        return c;
    }

    public static double[][][] reshape(double[][] x) {

        double[][][] out = new double[x.length][][];

        for (int i = 0; i < x.length; i++) {
            out[i] = reshape(x[i]);
        }

        return out;


    }

    public static double[][][][] reshapeBatch(double[][][] a, int bs) {


        if (a.length % bs != 0) {
            System.out.println("Error batch Size");
        }

        double[][][][] out = new double[a.length / bs][bs][a[0].length][a[0][0].length];


        int count = 0;
        double[][][] tmp = new double[bs][][];
        for (int i = 0; i < a.length % bs; i++) {

            for (int j = 0; j < bs; j++) {
                tmp[j] = a[count + j];
            }

            out[i] = tmp;
            count += bs;

        }

        return out;
    }


    public static void main(String[] args) throws Exception {


        double loss_per_epoch;
        double step_loss;
        double learning_rate = 0.01;
        int epochs = 20;
        String fpath = "./src/train_mnist.txt";
        double[][][] trainingData = MNIST.read(fpath, 60000);
        double[][] x_train = trainingData[0];
        double[][] y_train = trainingData[1];


        String fpath_test = "./src/test_mnist.txt";
        double[][][] testData = MNIST.read(fpath_test, 15000);
        double[][] x_test = testData[0];
        double[][] y_test = testData[1];

        double[][][] x_test_bs = reshape(x_test);


        double[][][] x_train_bs = reshape(x_train);
        double[][][][] x_train_bs4 = reshapeBatch(x_train_bs, 4);


        //x_train = null;
        //y_train = null;

        double[][] y_true;

        NeuralNetwork nn = new NeuralNetwork();
        Conv conv1D = new Conv(32);
        TanH act = new TanH();
        Conv1DFlatten cF = new Conv1DFlatten(8);

        MaxPooling2D pool = new MaxPooling2D();
        FullyConnectedLayer f1 = new FullyConnectedLayer(26 * 26 * 8, 20);
        TanH act2 = new TanH();

        FullyConnectedLayer f2 = new FullyConnectedLayer(20, 10);
        Softmax act3 = new Softmax();

        Losses loss = new MSE();

        int step_size = x_train.length;
        for (int i = 0; i < epochs; i++) {
            loss_per_epoch = 0;

            double[] outs;

            for (int j = 0; j < step_size; j++) {

                outs = cF.forward(x_train_bs[j]);
                outs = act.forward(outs);
                outs = f1.forward(outs);
                outs = act2.forward(outs);

                outs = f2.forward(outs);
                outs = act3.forward(outs);

                outs = Utils.clean_input(outs, y_train[0].length);
                step_loss = loss.forward(outs, y_train[j]);
                loss_per_epoch += step_loss;
                //calculates prime Loss
                outs = loss.backward(outs, y_train[j]);
                // now does back propagation // an updates the weights.
                outs = act3.backward(outs);
                outs = f2.backward(outs, learning_rate);

                outs = act2.backward(outs);
                outs = f1.backward(outs, learning_rate);
                outs = act.backward(outs);
                cF.backward(outs, learning_rate);


            }
            loss_per_epoch = loss_per_epoch / x_train.length;
            System.out.println("Loss per epoch: " + loss_per_epoch);


            int ist;
            int pred;
            double lossTest = 0;
            outs = null;
            for (int j = 0; j < x_test_bs.length; j++) {

                outs = cF.forward(x_test_bs[j]);
                outs = act.forward(outs);
                outs = f1.forward(outs);
                outs = act2.forward(outs);

                outs = f2.forward(outs);
                outs = act3.forward(outs);


                pred = Utils.argmax(outs);
                ist = Utils.argmax(y_test[j]);


                if (pred == ist) {
                    lossTest += 1;
                }

            }
            lossTest = lossTest / x_test_bs.length;
            System.out.println("Acc ing epoch: " + lossTest);

        }

    }


}




