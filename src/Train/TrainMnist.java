package Train;

import layer.FullyConnectedLayer;
import layer.NewSoftmax;
import layer.TanH;
import loss.MSE;
import main.MNIST;
import main.NeuralNetwork;
import optimizer.AdamNew;
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


        double[][][] trainingData = MNIST.read("data/mnist/train-images-idx3-ubyte.gz", "data/mnist/train-labels-idx1-ubyte.gz");
        double[][] x_train = trainingData[0];
        double[][] y_train = trainingData[1];


        double[][][] testData = MNIST.read("data/mnist/t10k-images-idx3-ubyte.gz", "data/mnist/t10k-labels-idx1-ubyte.gz");
        double[][] x_test = testData[0];
        double[][] y_test = testData[1];

        FullyConnectedLayer f1 = new FullyConnectedLayer(784, 80);
        FullyConnectedLayer f_out = new FullyConnectedLayer(80, 10);


        f1.genWeights(2);
        f_out.genWeights(2);

        System.out.println(getTop(new FullyConnectedLayer[]{f1, f_out}));


        f1.setActivation(new TanH());
        f1.setOptimizer(new AdamNew());
        f1.setUseBiases(false);
        //f2.setActivation(new TanH());
        f_out.setOptimizer(new AdamNew());
        f_out.setActivation(new TanH());
        f_out.setUseBiases(false);

        NewSoftmax act = new NewSoftmax();

        MSE loss = new MSE();

        double learning_rate = 1e-4;
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
                out = f_out.forward(f1.forward(out));
                //out = act.forward(out);

                loss_per_step += loss.forward(out, y_train[j]);
                out = loss.backward(out, y_train[j]);

                //out = act.backward(out);
                out = f_out.backward(out, learning_rate, i);
                //out = f2.backward(out, learning_rate);
                out = f1.backward(out, learning_rate, i);
            }

            System.out.println("Loss: " + loss_per_step / x_train.length);
            System.out.println("Time in sec. : " + ((System.currentTimeMillis() - st) / 1000));

            out = null;
            loss_per_step = 0;
            for (int ti = 0; ti < x_test.length; ti++) {

                out = Array_utils.copyArray(x_test[ti]);
                out = f1.forward(out);
                //out = f2.forward(out);
                out = f_out.forward(out);


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
            out = f_out.forward(out);


            if (Utils.argmax(out) == Utils.argmax(y_test[ti])) {
                loss_per_step += 1;

            }
        }
        System.out.println("Acc: " + loss_per_step / x_test.length);

        float acc = (float) loss_per_step / x_test.length;
        NeuralNetwork nn = new NeuralNetwork();
        FullyConnectedLayer[] layers = new FullyConnectedLayer[]{f1, f_out};
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
