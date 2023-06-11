package Train;

import layer.FullyConnectedLayer;
import layer.MSE;
import main.MNIST;
import utils.Array_utils;
import utils.Utils;

public class TrainMnistBatch {


    public static void main(String[] args) throws Exception {


        String fpath = "./src/train_mnist.txt";
        double[][][] trainingData = MNIST.read(fpath, 60000);
        double[][] x_train = trainingData[0];
        double[][] y_train = trainingData[1];

        String fpath_test = "./src/test_mnist.txt";
        double[][][] testData = MNIST.read(fpath_test, 15000);
        double[][] x_test = testData[0];
        double[][] y_test = testData[1];

        double[][][] x_train_bs = MNIST.x_train_2_batch(x_train, 4);
        double[][][] x_test_bs = MNIST.x_train_2_batch(x_test, 4);

        double[][][] y_train_bs = MNIST.y_train_2_batch(y_train, 4);
        double[][][] y_test_bs = MNIST.y_train_2_batch(y_test, 4);

        FullyConnectedLayer f1 = new FullyConnectedLayer(784, 40);
        FullyConnectedLayer f2 = new FullyConnectedLayer(40, 20);
        FullyConnectedLayer f3 = new FullyConnectedLayer(20, 10);

        MSE loss = new MSE();

        double learning_rate = 0.4;
        int epochs = 5;
        int step_size = x_train.length;

        //to validate after every Epoch.

        for (int i = 0; i < epochs; i++) {
            double[][] out;
            double loss_per_step = 0;
            for (int j = 0; j < step_size; j++) {

                out = Array_utils.copyArray(x_train_bs[j]);
                out = f1.forwardNew(out);
                out = f2.forwardNew(out);
                out = f3.forwardNew(out);


                loss_per_step += loss.forward(out, y_train_bs[j]);

                out = loss.backward(out, y_train_bs[j]);

                out = f3.backward(out, learning_rate);
                out = f2.backward(out, learning_rate);
                out = f1.backward(out, learning_rate);
            }

            System.out.println("Loss: " + loss_per_step / x_train.length);


        }

        double[][] out;
        double loss_per_step = 0;
        for (int ti = 0; ti < x_test.length; ti++) {

            out = x_test_bs[ti];
            out = f1.forwardNew(out);
            out = f2.forwardNew(out);
            out = f3.forwardNew(out);


            for (int bs = 0; bs < y_test_bs[0].length; bs++) {
                if (Utils.argmax(out[bs]) == Utils.argmax(y_test_bs[ti][bs])) {
                    loss_per_step += 1;

                }
            }


        }
        System.out.println("Acc: " + loss_per_step / x_test.length);


    }
}