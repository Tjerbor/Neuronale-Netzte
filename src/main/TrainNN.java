package main;

import layer.TanH;
import loss.MSE;
import optimizer.AdamNew;

import java.io.IOException;

public class TrainNN {


    public static void main(String[] args) throws IOException {


        NN_New nn = new NN_New();

        nn.add(new FullyConnectedLayerNew(784, 40, new TanH()));
        nn.add(new FullyConnectedLayerNew(40, 10, new TanH()));

        double[][][] trainingData = MNIST.read("data/mnist/train-images-idx3-ubyte.gz", "data/mnist/train-labels-idx1-ubyte.gz");
        double[][] x_train = trainingData[0];
        double[][] y_train = trainingData[1];


        double[][][] testData = MNIST.read("data/mnist/t10k-images-idx3-ubyte.gz", "data/mnist/t10k-labels-idx1-ubyte.gz");
        double[][] x_test = testData[0];
        double[][] y_test = testData[1];


        nn.setLoss(new MSE());
        nn.setOptimizer(new AdamNew());

        nn.train(10, x_train, y_train, x_test, y_test);


    }
}
