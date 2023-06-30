package Train;

import builder.NetworkBuilder;
import loss.MSE;
import main.MNIST;
import main.NeuralNetwork;

import java.io.IOException;

public class TrainStats {


    public static void main(String[] args) throws IOException {


        double[][][] trainingData = MNIST.read("data/mnist/train-images-idx3-ubyte.gz", "data/mnist/train-labels-idx1-ubyte.gz");
        double[][] x_train = trainingData[0];
        double[][] y_train = trainingData[1];


        double[][][] testData = MNIST.read("data/mnist/t10k-images-idx3-ubyte.gz", "data/mnist/t10k-labels-idx1-ubyte.gz");
        double[][] x_test = testData[0];
        double[][] y_test = testData[1];

        NetworkBuilder builder = new NetworkBuilder();
        builder.addOnlyFastLayer(new int[]{784, 49, 20, 10});


        NeuralNetwork nn = builder.getModel();

        nn.setLoss(new MSE());
        nn.trainPerStep(300000, 10000, x_train, y_train, x_test, y_test, 0.01);
        nn.writeModel("nn_weights.txt");


    }
}
