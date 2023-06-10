package Train;

import layer.*;
import main.MNIST;

import static Train.TrainMnistConv.reshape;
import static main.MNIST.y_train_2_batch;
import static utils.Array_utils.getShape;

public class TrainMnistFull {

    static int[] shape;

    public static double[][][] zerosLike(int[] a) {
        return new double[a[0]][a[1]][a[2]];

    }

    public static double[][][][] zerosLikeBatch(int[] a) {
        return new double[a[0]][a[1]][a[2]][a[3]];

    }

    public static double[][][] reFlat(double[] a) {
        double[][][] b = zerosLike(shape);
        int c = 0;
        for (int i = 0; i < b.length; i++) {
            for (int j = 0; j < b[0].length; j++) {
                for (int k = 0; k < b[0][0].length; k++) {
                    b[i][j][k] = a[c];
                    c += 1;
                }


            }
        }

        TrainMnistFull.shape = getShape(a);

        return b;
    }

    public static double[][][][] reFlat(double[][] a) {
        double[][][][] b = zerosLikeBatch(shape);
        for (int i = 0; i < a.length; i++) {
            b[i] = reFlat(a[i]);
        }

        TrainMnistFull.shape = getShape(a);

        return b;
    }

    public static double[] flatten(double[][][] a) {
        double[] b = new double[a.length * a[0].length * a[0][0].length];
        int c = 0;
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                for (int k = 0; k < a[0][0].length; k++) {
                    b[c] = a[i][j][k];
                    c += 1;
                }


            }
        }

        TrainMnistFull.shape = getShape(a);

        return b;
    }

    public static double[][] flatten(double[][][][] a) {
        double[][] b = new double[a.length][a[0].length * a[0][0].length * a[0][0][0].length];
        int c = 0;
        for (int i = 0; i < a.length; i++) {
            b[i] = flatten(a[i]);
        }

        TrainMnistFull.shape = getShape(a);

        return b;
    }

    public static double[][][][] reshapeBatch(double[][][] a, int bs) {


        if (a.length % bs != 0) {
            System.out.println("Error batch Size");
        }

        double[][][][] out = new double[(int) (a.length / bs)][bs][a[0].length][a[0][0].length];


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
        double[][][][] x_train_bs4 = reshapeBatch(reshape(x_train), 4);
        double[][][] y_train_bs4 = y_train_2_batch(y_train, 4);


        x_train = null;
        //y_train = null;
        x_train_bs = null;

        double[][] y_true;

        Conv conv1D = new Conv(8);
        TanH act = new TanH();
        MaxPooling2D poll = new MaxPooling2D();

        FullyConnectedLayer f1 = new FullyConnectedLayer(13 * 13 * 8, 20);
        TanH act2 = new TanH();

        FullyConnectedLayer f2 = new FullyConnectedLayer(20, 10);
        Softmax act3 = new Softmax();

        Losses loss = new MSE();

        Flatten flatt = new Flatten(false);

        int step_size = x_train.length;
        for (int i = 0; i < epochs; i++) {
            loss_per_epoch = 0;

            double[][] outs;
            double[][][][] tmp;

            for (int j = 0; j < step_size; j++) {


                tmp = conv1D.forward(x_train_bs4[j]);
                tmp = act.forward(tmp);
                tmp = poll.forward(tmp);

                outs = flatten(tmp);
                outs = f1.forward(outs);
                outs = act2.forward(outs);

                outs = f2.forward(outs);
                outs = act3.forward(outs);

                //outs = Utils.clean_input(outs, y_train[0].length);
                step_loss = loss.forward(outs, y_train_bs4[j]);
                loss_per_epoch += step_loss;
                //calculates prime Loss

                outs = loss.backward(outs, y_train_bs4[j]);
                // now does back propagation // an updates the weights.
                outs = act3.backward(outs);
                outs = f2.backward(outs, learning_rate);

                outs = act2.backward(outs);
                outs = f1.backward(outs, learning_rate);
                outs = act.backward(outs);

                tmp = reFlat(outs);

                tmp = poll.backward(tmp);
                conv1D.backward(tmp, learning_rate);


            }
            loss_per_epoch = loss_per_epoch / x_train.length;
            System.out.println("Loss per epoch: " + loss_per_epoch);
        }
    }
}
