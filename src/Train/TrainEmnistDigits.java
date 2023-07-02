package Train;

import builder.NetworkBuilder;
import function.TanH;
import loss.MSE;
import main.MNIST;
import main.NeuralNetwork;

import java.io.IOException;

public class TrainEmnistDigits {


    public static void main(String[] args) throws IOException {


        int numClasses = 10;
        double[][][] trainingData = MNIST.read("data/emnist/emnist-digits-train-images-idx3-ubyte.gz", "data/emnist/emnist-digits-train-labels-idx1-ubyte.gz", numClasses);
        double[][] x_train = trainingData[0];
        double[][] y_train = trainingData[1];


        double[][][] testData =
                MNIST.read("data/emnist/emnist-digits-test-images-idx3-ubyte.gz", "data/emnist/emnist-digits-test-labels-idx1-ubyte.gz", numClasses);
        double[][] x_test = testData[0];
        double[][] y_test = testData[1];

        NetworkBuilder builder = new NetworkBuilder();
        builder.addOnlyFCL(new int[]{784, 80, 40, numClasses}, new TanH());
        
        NeuralNetwork nn = builder.getModel();
        nn.printSummary();

        System.out.println(x_train.length);
        nn.setLoss(new MSE());
        nn.train(3, x_train, y_train, x_test, y_test, 0.4);
        nn.writeModel("nn_emnist_digits_weights.txt");

    }
}
