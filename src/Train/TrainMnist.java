package Train;

import layer.*;
import main.MNIST;
import utils.Array_utils;
import utils.Utils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class TrainMnist {


    public static void export(String filename, FullyConnectedLayer[] layers) {

        StringBuilder s = new StringBuilder();
        s.append("Topology: ").append("layers;784,49,28,10").append("\n");
        for (FullyConnectedLayer layer : layers) {

            s.append(layer.toString(true));
            s.append("\n");
            s.append("\n");


        }

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            writer.write(s.toString());
            System.out.println("Wrote File");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }


    }


    public static void main(String[] args) throws Exception {


        String fpath = "/home/dblade/Documents/Neuronale-Netzte/src/train_mnist.txt";
        double[][][] trainingData = MNIST.read(fpath, 60000);
        double[][] x_train = trainingData[0];
        double[][] y_train = trainingData[1];


        String fpath_test = "/home/dblade/Documents/Neuronale-Netzte/src/test_mnist.txt";
        double[][][] testData = MNIST.read(fpath_test, 15000);
        double[][] x_test = testData[0];
        double[][] y_test = testData[1];

        FullyConnectedLayer f1 = new FullyConnectedLayer(784, 40);
        //FullyConnectedLayer f2 = new FullyConnectedLayer(40, 20);
        FullyConnectedLayer f3 = new FullyConnectedLayer(40, 10);


        f1.setActivation(new TanH());
        //f2.setActivation(new ReLu());
        f3.setActivation(new Activation());

        Softmax act = new Softmax();

        MSE loss = new MSE();

        double learning_rate = 0.3;
        int epochs = 7;
        int step_size = x_train.length;

        //to validate after every Epoch.
        long st;
        for (int i = 0; i < epochs; i++) {
            st = System.currentTimeMillis();
            double[] out;
            double loss_per_step = 0;

            //learning_rate -= (learning_rate * 0.05);
            for (int j = 0; j < step_size; j++) {

                out = Array_utils.copyArray(x_train[j]);
                out = f1.forward(out);
                //out = f2.forward(out);
                out = f3.forward(out);
                out = act.forward(out);

                loss_per_step += loss.forward(out, y_train[j]);
                out = loss.backward(out, y_train[j]);

                out = act.backward(out);
                out = f3.backward(out, learning_rate);
                //out = f2.backward(out, learning_rate);
                out = f1.backward(out, learning_rate);
            }

            System.out.println("Loss: " + loss_per_step / x_train.length);
            System.out.println("Time in sec. : " + ((System.currentTimeMillis() - st) / 1000));

            out = null;
            loss_per_step = 0;
            for (int ti = 0; ti < x_test.length; ti++) {

                out = x_test[ti];
                out = f1.forward(out);
                //out = f2.forward(out);
                out = f3.forward(out);


                if (Utils.argmax(out) == Utils.argmax(y_test[ti])) {
                    loss_per_step += 1;

                }
            }
            System.out.println("Acc: " + loss_per_step / x_test.length);


        }


        double[] out;
        double loss_per_step = 0;
        for (int ti = 0; ti < x_test.length; ti++) {

            out = x_test[ti];
            out = f1.forward(out);
            //out = f2.forward(out);
            out = f3.forward(out);


            if (Utils.argmax(out) == Utils.argmax(y_test[ti])) {
                loss_per_step += 1;

            }
        }
        System.out.println("Acc: " + loss_per_step / x_test.length);
        FullyConnectedLayer[] layers = new FullyConnectedLayer[]{f1, f3};
        String outFPath = "weights_" + loss_per_step / x_test.length + ".txt";
        export(outFPath, layers);

    }


}