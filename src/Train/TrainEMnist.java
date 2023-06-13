package Train;

import layer.FullyConnectedLayer;
import layer.Losses;
import layer.MSE;
import main.EMNIST;
import utils.Array_utils;
import utils.Utils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class TrainEMnist {


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


        String type_ = "balanced";
        //bymerge maxSize 697932
        //balanced maxSize 112800
        String fpath = "./src/TrainData/train_emnist_" + type_ + ".txt";
        double[][][] trainingData = EMNIST.read(fpath, 112800);
        double[][] x_train = trainingData[0];
        double[][] y_train = trainingData[1];


        FullyConnectedLayer f1 = new FullyConnectedLayer(784, 120, "tanh");
        FullyConnectedLayer f2 = new FullyConnectedLayer(120, 60, "tanh");
        FullyConnectedLayer f3 = new FullyConnectedLayer(60, 47, "tanh");

        //LayerNorm1D l1 = new LayerNorm1D(120);

        f1.activateMomentum();
        f1.activateBiases();

        f2.activateMomentum();
        f2.activateBiases();

        f2.activateMomentum();
        f2.activateBiases();


        Losses loss = new MSE();

        double learning_rate = 0.4;
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
                out = f1.forward(out);
                out = f2.forward(out);
                //out = l1.forward(out);
                out = f3.forward(out);


                loss_per_step += loss.forward(out, y_train[j]);

                out = loss.backward(out, y_train[j]);

                out = f3.backward(out, learning_rate);
                //out = l1.backward(out);
                out = f2.backward(out, learning_rate);
                out = f1.backward(out, learning_rate);
            }

            System.out.println("Loss: " + loss_per_step / x_train.length);
            System.out.println("Time in sec. : " + ((System.currentTimeMillis() - st) / 1000));


        }
        // bymerge maxSize 116323
        // balanced maxSize 18800
        String fpath_test = "./src/TrainData/test_emnist_" + type_ + ".txt";

        int maxSizeTest = 18800;
        if (type_.equals("bymerge")) {
            maxSizeTest = 116323;
        }


        double[][][] testData = EMNIST.read(fpath_test, maxSizeTest);
        double[][] x_test = testData[0];
        double[][] y_test = testData[1];

        double[] out;
        double loss_per_step = 0;
        for (int ti = 0; ti < x_test.length; ti++) {

            out = x_test[ti];
            out = f1.forward(out);
            out = f2.forward(out);
            out = f3.forward(out);


            if (Utils.argmax(out) == Utils.argmax(y_test[ti])) {
                loss_per_step += 1;

            }
        }
        System.out.println("Acc: " + loss_per_step / x_test.length);


        FullyConnectedLayer[] layers = new FullyConnectedLayer[]{f1, f2, f3};
        String outFPath = "weights_" + loss_per_step / x_test.length + ".txt";
        export(outFPath, layers);

    }


}