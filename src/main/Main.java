package main;

import load.LoadModel;
import utils.ImageReader;

import java.io.IOException;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) throws IOException {
        mnist();
        emnist();
    }

    public static void mnist() throws IOException {
        NeuralNetwork neuralNetwork = LoadModel.loadModel("mnist.csv");

        double[][][] training = MNIST.read("data/mnist/train-images-idx3-ubyte.gz", "data/mnist/train-labels-idx1-ubyte.gz");
        double[][][] test = MNIST.read("data/mnist/t10k-images-idx3-ubyte.gz", "data/mnist/t10k-labels-idx1-ubyte.gz");

        neuralNetwork.printTestStats(training[0], training[1]);
        neuralNetwork.printTestStats(test[0], test[1]);
    }

    public static void emnist() throws IOException {
        NeuralNetwork neuralNetwork = LoadModel.loadModel("mnist.csv");

        double[][][] training = MNIST.read("data/emnist/emnist-letters-train-images-idx3-ubyte.gz", "data/emnist/emnist-letters-train-labels-idx1-ubyte.gz", 26);
        double[][][] test = MNIST.read("data/emnist/emnist-letters-test-images-idx3-ubyte.gz", "data/emnist/emnist-letters-test-labels-idx1-ubyte.gz", 26);

        neuralNetwork.printTestStats(training[0], training[1]);
        neuralNetwork.printTestStats(test[0], test[1]);
    }

    public static void ImageOnTheFLyTest() throws IOException {
        //TODO
        NeuralNetwork neuralNetwork = LoadModel.loadModel(null);

        String path = "uwu.png";
        double[] testData = ImageReader.ImageToArray(path);

        double[] probabilities = neuralNetwork.compute(testData);

        System.out.println(Arrays.toString(probabilities));
    }


}
