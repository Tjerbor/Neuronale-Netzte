package Train;

import builder.NetworkBuilder;
import function.TanH;
import loss.MSE;
import main.MNIST;
import main.NeuralNetwork;

import java.io.IOException;

public class TrainEmnistBalanced {


    public static void main(String[] args) throws IOException {


        int numClasses = 47;
        double[][][] trainingData = MNIST.readBalanced("data/emnist/emnist-balanced-train-images-idx3-ubyte.gz", "data/emnist/emnist-balanced-train-labels-idx1-ubyte.gz", numClasses);
        double[][] x_train = trainingData[0];
        double[][] y_train = trainingData[1];


        double[][][] testData = MNIST.readBalanced("data/emnist/emnist-balanced-test-images-idx3-ubyte.gz", "data/emnist/emnist-balanced-test-labels-idx1-ubyte.gz", numClasses);
        double[][] x_test = testData[0];
        double[][] y_test = testData[1];

        NetworkBuilder builder = new NetworkBuilder();
        builder.addOnlyFCL(new int[]{784, 25 * 25, 15 * 15, 10 * 10, numClasses * 4, numClasses}, new TanH(), true);

        NeuralNetwork nn = builder.getModel();

        System.out.println("Train-Data Size: " + x_train.length + " : " + y_train.length);
        nn.setLoss(new MSE());
        nn.train(10, x_train, y_train, x_test, y_test, 0.4);
        nn.writeModel("nn_emnist_balanced_weights.txt");

    }
}
