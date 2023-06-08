package Train;

import layer.*;
import main.Mnist_reader;
import main.NeuralNetwork;
import utils.Utils;

import static Train.TrainMnistConv.reshape;

public class TrainMnistFull {


    public static void main(String[] args) throws Exception {


        double loss_per_epoch;
        double step_loss;
        double learning_rate = 0.01;
        int epochs = 20;
        Mnist_reader.limit = 60000;
        String fpath = "./src/train_mnist.txt";
        double[][] x_train = Mnist_reader.getTrainData_x(fpath);
        double[][] y_train = Mnist_reader.getTrainData_y(fpath);


        String fpath_test = "./src/test_mnist.txt";
        Mnist_reader.limit = 15000;
        double[][] x_test = Mnist_reader.getTrainData_x(fpath);
        double[][] y_test = Mnist_reader.getTrainData_y(fpath);

        double[][][] x_test_bs = reshape(x_test);

        double[][][] x_train_bs = reshape(x_train);


        //x_train = null;
        //y_train = null;

        double[][] y_true;

        NeuralNetwork nn = new NeuralNetwork();
        Conv conv1D = new Conv(8);
        TanH act = new TanH();
        Conv1DFlatten cF = new Conv1DFlatten(8);

        FullyConnectedLayer f1 = new FullyConnectedLayer(13 * 13 * 8, 20);
        TanH act2 = new TanH();

        FullyConnectedLayer f2 = new FullyConnectedLayer(20, 10);
        Softmax act3 = new Softmax();

        Losses loss = new MSE();

        Flatten flatt = new Flatten(true);

        int step_size = x_train.length;
        for (int i = 0; i < epochs; i++) {
            loss_per_epoch = 0;

            double[] outs;
            double[][][] tmp;

            for (int j = 0; j < step_size; j++) {


                tmp = conv1D.forward(x_train_bs[j]);
                tmp = act.forward(tmp);


                outs = (double[]) flatt.forward(tmp);
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

                tmp = (double[][][]) flatt.backward(outs);

                cF.backward(outs, learning_rate);


            }
            loss_per_epoch = loss_per_epoch / x_train.length;
            System.out.println("Loss per epoch: " + loss_per_epoch);
        }
    }
}
