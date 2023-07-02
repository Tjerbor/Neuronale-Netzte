package examples;

import builder.NetworkBuilder;
import function.TanH;
import load.LoadModel;
import loss.MSE;
import main.MNIST;
import main.NeuralNetwork;

import java.io.IOException;

import static utils.TrainUtils.reshapeToChannelsLast;

public class ExpConv2D {

    public static void main(String[] args) throws IOException {

        int numFilter = 8;
        int numClasses = 10;
        int[] inputShape = new int[]{1, 28, 28};
        int kernelSize = 5;
        int strides = 1; //stepSize.

        NetworkBuilder builder = new NetworkBuilder(inputShape);

        //builder.addConv2D(numFilter, kernelSize, strides);
        //builder.addDropout(0.1);
        //builder.addActivationLayer(new ReLu());

        builder.addConv2D(numFilter, kernelSize, strides);
        builder.addDropout(0.1);
        builder.addActivationLayer(new TanH());

        builder.addMaxPooling2D(2, 4); //uses for standard strides2 and poolSize2

        builder.addFlatten();
        builder.addFCL(numClasses);
        builder.addActivationLayer(new TanH());

        NeuralNetwork nn = builder.getModel();
        nn.printSummary();

        double[][][] trainingData = MNIST.read("data/mnist/train-images-idx3-ubyte.gz", "data/mnist/train-labels-idx1-ubyte.gz");
        double[][] x_train = trainingData[0];
        double[][] y_train = trainingData[1];


        double[][][] testData = MNIST.read("data/mnist/t10k-images-idx3-ubyte.gz", "data/mnist/t10k-labels-idx1-ubyte.gz");
        double[][] x_test = testData[0];
        double[][] y_test = testData[1];

        double[][][][] x_convT = new double[x_train.length][][][];
        double[][][][] x_convTest = new double[x_test.length][][][];


        for (int i = 0; i < x_train.length; i++) {
            x_convT[i] = reshapeToChannelsLast(x_train[i], new int[]{1, 28, 28});
        }

        for (int i = 0; i < x_test.length; i++) {
            x_convTest[i] = reshapeToChannelsLast(x_test[i], new int[]{1, 28, 28});
        }


        nn.setLoss(new MSE());

        //nn = LoadModel.loadModel("nn_weights_conv2.txt");
        nn.trainTesting(7, x_convT, y_train, x_convTest, y_test, 0.1);
        nn.writeModel("nn_weights_conv4.txt");
        nn = LoadModel.loadModel("nn_weights_conv4.txt");
        nn.printSummary();


    }
}
