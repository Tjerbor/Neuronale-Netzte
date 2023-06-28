package Train;

import load.LoadModel;
import main.MNIST;
import main.NeuralNetwork;

import java.io.IOException;

public class TestData {

    public static void main(String[] args) throws IOException {

        NeuralNetwork nn = LoadModel.loadModel("nn_weights.txt");

        double[][][] trainingData = MNIST.read("data/mnist/train-images-idx3-ubyte.gz", "data/mnist/train-labels-idx1-ubyte.gz");
        double[][] x_train = trainingData[0];
        double[][] y_train = trainingData[1];


        double[][][] testData = MNIST.read("data/mnist/t10k-images-idx3-ubyte.gz", "data/mnist/t10k-labels-idx1-ubyte.gz");
        double[][] x_test = testData[0];
        double[][] y_test = testData[1];

        nn.printTestStats(x_train, y_train);
        nn.printTestStats(x_test, y_test);


    }
}
