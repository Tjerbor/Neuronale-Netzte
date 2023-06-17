package Train;

import layer.FullyConnectedLayer;
import layer.TanH;
import utils.Reader;
import utils.Utils;

import java.io.IOException;

import static Train.LoadOwn.getTestData;
import static Train.LoadOwn.loadSingle;

public class TestOwnData {


    public static void main(String[] args) throws IOException {


        String weights = "/home/dblade/Documents/Neuronale-Netzte/weights_0.9605333_.txt";
        FullyConnectedLayer[] layers = Reader.create(weights);
        String dirFpath = "/home/dblade/Documents/Neuronale-Netzte/src/Train/OwnData";
        double[][][] testData = getTestData(dirFpath);

        for (int i = 0; i < layers.length; i++) {
            layers[i].setActivation(new TanH());
        }

        double[][] y_test = testData[1];
        double[][] x_test = testData[0];


        double[][] test = new double[x_test.length][784];

        for (int i = 0; i < x_test.length; i++) {
            for (int j = 0; j < x_test[0].length; j++) {
                test[i][j] = x_test[i][783 - j];
            }
        }


        double[] out = null;
        double loss_per_step = 0;
        for (int ti = 0; ti < x_test.length; ti++) {


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

        double[][] data = loadSingle("/home/dblade/Documents/Neuronale-Netzte/src/pythonScripts/test_0.txt");

        out = data[0];
        for (int i = 0; i < layers.length; i++) {
            out = layers[i].forward(out);
        }

        System.out.println("Predicted Class: " + Utils.argmax(out));
        System.out.println("Actual Class: " + Utils.argmax(data[1]));
        System.out.println("");

        if (Utils.argmax(out) == Utils.argmax(data[1])) {
            loss_per_step += 1;

        }

    }
}
