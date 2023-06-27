package Train;

import builder.BuildNetwork;
import loss.MSE;
import main.MNIST;
import main.NeuralNetwork;
import optimizer.AdamNew;

import java.io.IOException;

public class TrainBuilder {

    public static void main(String[] args) throws IOException {


        int[] topologie = new int[]{784, 20, 10};
        BuildNetwork builder = new BuildNetwork(topologie[0]);

        for (int i = 1; i < topologie.length; i++) {
            builder.addFastLayer(topologie[i]);
        }

        NeuralNetwork nn = builder.getModel();

        nn.printSummary();

        double[][][] trainingData = MNIST.read("data/mnist/train-images-idx3-ubyte.gz", "data/mnist/train-labels-idx1-ubyte.gz");
        double[][] x_train = trainingData[0];
        double[][] y_train = trainingData[1];


        double[][][] testData = MNIST.read("data/mnist/t10k-images-idx3-ubyte.gz", "data/mnist/t10k-labels-idx1-ubyte.gz");
        double[][] x_test = testData[0];
        double[][] y_test = testData[1];

        int epochs = 5;
        double learningRate = 0.3;

        nn.setOptimizer(new AdamNew());
        nn.setLoss(new MSE());
        nn.train(epochs, x_train, y_train, x_test, y_test, learningRate);


    }
}
