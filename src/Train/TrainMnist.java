package Train;

import extraLayer.FastLinearLayer;
import layer.FullyConnectedLayer;
import layer.TanH;
import loss.MSE;
import main.MNIST;
import main.NeuralNetwork;
import utils.Array_utils;
import utils.Utils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import static Train.LoadOwn.getTestData;
import static utils.TrainUtils.shuffle;

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


        double[][][] trainingData = MNIST.read("data/mnist/train-images-idx3-ubyte.gz", "data/mnist/train-labels-idx1-ubyte.gz");
        double[][] x_train = trainingData[0];
        double[][] y_train = trainingData[1];


        double[][][] testData = MNIST.read("data/mnist/t10k-images-idx3-ubyte.gz", "data/mnist/t10k-labels-idx1-ubyte.gz");
        double[][] x_test = testData[0];
        double[][] y_test = testData[1];

        double learning_rate = 0.3;
        FastLinearLayer f1 = new FastLinearLayer(784, 49, learning_rate);
        FastLinearLayer f_out = new FastLinearLayer(49, 10, learning_rate);

        f1.setNextLayer(f_out);
        f_out.setPreviousLayer(f1);

        MSE loss = new MSE();


        int epochs = 7;
        int step_size = x_train.length;

        //to validate after every Epoch.
        long st;
        for (int i = 0; i < epochs; i++) {

            trainingData = shuffle(x_train, y_train);

            double[][] trainX = trainingData[0];
            double[][] trainY = trainingData[1];

            st = System.currentTimeMillis();
            double[] out;
            double loss_per_step = 0;

            //learning_rate -= (learning_rate * 0.05);
            for (int j = 0; j < step_size; j++) {

                out = Array_utils.copyArray(trainX[j]);

                f1.forward(out);
                out = f_out.getOutput().getData1D();
                //out = act.forward(out);

                loss_per_step += loss.forward(out, trainY[j]);
                out = loss.backward(out, trainY[j]);

                //out = act.backward(out);
                f_out.backward(out);

            }

            System.out.println("Loss: " + loss_per_step / x_train.length);
            System.out.println("Time in sec. : " + ((System.currentTimeMillis() - st) / 1000));

            out = null;
            loss_per_step = 0;
            for (int ti = 0; ti < x_test.length; ti++) {

                out = Array_utils.copyArray(x_test[ti]);
                f1.forward(out);
                out = f_out.getOutput().getData1D();


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
            f1.forward(out);
            out = f_out.getOutput().getData1D();


            if (Utils.argmax(out) == Utils.argmax(y_test[ti])) {
                loss_per_step += 1;

            }
        }
        System.out.println("Acc: " + loss_per_step / x_test.length);

        float acc = (float) loss_per_step / x_test.length;
        NeuralNetwork nn = new NeuralNetwork();
        FullyConnectedLayer[] layers = new FullyConnectedLayer[2];
        nn.setLayers(layers);
        nn.exportWeights("weights_" + acc + "_.txt");
        String outFPath = "weights_" + loss_per_step / x_test.length + ".txt";
        export(outFPath, layers);


        //test own Data
        String dirFpath = "./src/Train/OwnData";
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
