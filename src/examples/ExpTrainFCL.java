package examples;

import builder.NetworkBuilder;
import function.TanH;
import loss.MSE;
import main.MNIST;
import main.NeuralNetwork;

import java.io.IOException;

public class ExpTrainFCL {

    public static void main(String[] args) throws IOException {


        /*
        NetworkBuilder builder = new NetworkBuilder(784);
        builder.addFCL(40, false);
        builder.addActivationLayer(new TanH());
        builder.addFCL(20, false);
        builder.addActivationLayer(new TanH());
        builder.addFCL(10, false);
        builder.addActivationLayer(new TanH());
        NeuralNetwork nn = builder.getModel();
         */

        int numClasses = 26;
        double[][][] trainingData = MNIST.read("data/emnist/emnist-letters-train-images-idx3-ubyte.gz", "data/emnist/emnist-letters-train-labels-idx1-ubyte.gz", numClasses);


        double[][][] testData = MNIST.read("data/emnist/emnist-letters-test-images-idx3-ubyte.gz", "data/emnist/emnist-letters-test-labels-idx1-ubyte.gz", numClasses);

        NetworkBuilder builder = new NetworkBuilder();
        int[] topologie = new int[]{784, 20 * 20, 10 * 10, numClasses};
        builder.addOnlyFCL(topologie, new TanH(), true);


        double learningRate = 0.4;


        NeuralNetwork nn = builder.getModel();

        nn.printSummary();

        nn.setLoss(new MSE());
        nn.train(20, trainingData[0], trainingData[1], testData[0], testData[1], learningRate);
        nn.writeModel("weights.csv");

    }

}
