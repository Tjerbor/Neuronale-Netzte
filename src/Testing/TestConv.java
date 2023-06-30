package Testing;

import builder.NetworkBuilder;
import function.TanH;
import loss.Loss;
import loss.MSE;
import main.MNIST;
import main.NeuralNetwork;
import utils.Matrix;

import java.io.IOException;
import java.util.Arrays;

public class TestConv {


    public static double[][][] reshapeChannelFirst(double[] a) {


        int count = 0;
        double[][][] c = new double[1][28][28];
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                c[0][i][j] = a[count];
                count += 1;
            }
        }

        return c;
    }

    public static void main(String[] args) throws IOException {


        int numFilter = 8;
        int numClasses = 10;
        int[] inputShape = new int[]{1, 28, 28};
        int kernelSize = 5;
        int strides = 1; //stepSize.

        NetworkBuilder builder = new NetworkBuilder(inputShape);
        builder.addConv2D(numFilter, kernelSize, strides);
        //builder.addDropout(0.5);
        builder.addActivationLayer(new TanH());
        builder.addMaxPooling2D(2, 2); //uses for standard strides2 and poolSize2
        builder.addFlatten();
        builder.addFCL(numClasses);
        builder.addActivationLayer(new TanH());

        NeuralNetwork nn = builder.getModel();
        nn.printSummary();
        nn.printSummaryBackward();

        double[][][] trainingData = MNIST.read("data/mnist/train-images-idx3-ubyte.gz", "data/mnist/train-labels-idx1-ubyte.gz");
        double[][] x_train = trainingData[0];
        double[][] y_train = trainingData[1];


        double[][][] testData = MNIST.read("data/mnist/t10k-images-idx3-ubyte.gz", "data/mnist/t10k-labels-idx1-ubyte.gz");
        double[][] x_test = testData[0];
        double[][] y_test = testData[1];

        double[][][][] x_convT = new double[x_train.length][][][];
        double[][][][] x_convTest = new double[x_test.length][][][];


        for (int i = 0; i < x_train.length; i++) {
            x_convT[i] = reshapeChannelFirst(x_train[i]);
        }

        for (int i = 0; i < x_test.length; i++) {
            x_convTest[i] = reshapeChannelFirst(x_test[i]);
        }

        nn.getInputLayer().forward(new Matrix(x_convT[0]));

        Matrix m = nn.getOutputLayer().getOutput();

        Loss loss = new MSE();
        System.out.println(Arrays.toString(m.getData1D()));
    }
}
