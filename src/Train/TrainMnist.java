package Train;

import layer.FullyConnectedLayer;
import layer.MSE;
import layer.NewSoftmax;
import layer.TanH;
import main.MNIST;
import main.NeuralNetwork;
import utils.Array_utils;
import utils.Utils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import static Train.LoadOwn.getTestData;

public class TrainMnist {


    public static void export(String filename, FullyConnectedLayer[] layers) {

        StringBuilder s = new StringBuilder();
        s.append(getTop(layers)).append("\n");
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

        FullyConnectedLayer f1 = new FullyConnectedLayer(784, 49);
        FullyConnectedLayer f3 = new FullyConnectedLayer(49, 10);


        System.out.println(getTop(new FullyConnectedLayer[]{f1, f3}));

        f1.setActivation(new TanH());
        //f1.setOptimizer(new RMSPropNew());
        //f2.setActivation(new TanH());
        //f3.setOptimizer(new RMSPropNew());
        f3.setActivation(new TanH());

        NewSoftmax act = new NewSoftmax();

        MSE loss = new MSE();

        double learning_rate = 0.1;
        int epochs = 7;
        int step_size = x_train.length;

        //to validate after every Epoch.
        long st;
        for (int i = 0; i < epochs; i++) {
            st = System.currentTimeMillis();
            double[] out;
            double loss_per_step = 0;

            learning_rate -= (learning_rate * 0.05);
            for (int j = 0; j < step_size; j++) {

                out = Array_utils.copyArray(x_train[j]);
                out = f3.forward(f1.forward(out));
                //out = act.forward(out);

                loss_per_step += loss.forward(out, y_train[j]);
                out = loss.backward(out, y_train[j]);

                //out = act.backward(out);
                out = f3.backward(out, learning_rate);
                //out = f2.backward(out, learning_rate);
                out = f1.backward(out, learning_rate);
            }

            System.out.println("Loss: " + loss_per_step / x_train.length);
            System.out.println("Time in sec. : " + ((System.currentTimeMillis() - st) / 1000));

            out = null;
            loss_per_step = 0;
            for (int ti = 0; ti < x_test.length; ti++) {

                out = Array_utils.copyArray(x_test[ti]);
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

        float acc = (float) loss_per_step / x_test.length;
        NeuralNetwork nn = new NeuralNetwork();
        FullyConnectedLayer[] layers = new FullyConnectedLayer[]{f1, f3};
        nn.setLayers(layers);
        nn.exportWeights("weights_" + acc + "_.txt");
        String outFPath = "weights_" + loss_per_step / x_test.length + ".txt";
        export(outFPath, layers);


        //test own Data
        String dirFpath = "/home/dblade/Documents/Neuronale-Netzte/src/Train/OwnData";
        testData = getTestData(dirFpath);

        for (int i = 0; i < layers.length; i++) {
            layers[i].setActivation(new TanH());
        }

        y_test = testData[1];
        x_test = testData[0];


        double[][] test = new double[x_test.length][784];

        for (int i = 0; i < x_test.length; i++) {
            for (int j = 0; j < x_test[0].length; j++) {
                test[i][j] = x_test[i][783 - j];
            }
        }


        out = null;
        loss_per_step = 0;
        for (int ti = 0; ti < x_test.length; ti++) {


            out = x_test[ti];
            for (int i = 0; i < layers.length; i++) {
                out = layers[i].forward(out);
            }

            System.out.println("Predicted Class: " + Utils.argmax(out));
            System.out.println("Actual Class: " + Utils.argmax(y_test[ti]));
            System.out.println("");


            out = test[ti];
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


    }

    public static String getTop(FullyConnectedLayer[] layers) {

        int[] shape = new int[layers.length + 1];

        for (int i = 0; i < layers.length; i++) {
            shape[i] = layers[i].getWeights().length - 1;
            shape[i + 1] = layers[i].getBiases().length;
        }

        String s = "layers;";
        for (int i = 0; i < shape.length; i++) {

            if (i != shape.length - 1) {
                s += shape[i] + ";";

            } else {
                s += shape[i];
            }
        }

        return s;
    }


}