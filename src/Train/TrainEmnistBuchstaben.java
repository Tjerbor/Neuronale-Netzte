package Train;

import builder.NetworkBuilder;
import function.TanH;
import load.LoadModel;
import loss.MSE;
import main.MNIST;
import main.NeuralNetwork;

import java.io.IOException;
import java.util.Arrays;

public class TrainEmnistBuchstaben {


    public static void main(String[] args) throws IOException {


        int numClasses = 26;
        double[][][] trainingData = MNIST.read("data/emnist/emnist-letters-train-images-idx3-ubyte.gz", "data/emnist/emnist-letters-train-labels-idx1-ubyte.gz", numClasses);
        double[][] x_train = trainingData[0];
        double[][] y_train = trainingData[1];


        double[][][] testData = MNIST.read("data/emnist/emnist-letters-test-images-idx3-ubyte.gz", "data/emnist/emnist-letters-test-labels-idx1-ubyte.gz", numClasses);
        double[][] x_test = testData[0];
        double[][] y_test = testData[1];

        NetworkBuilder builder = new NetworkBuilder();
        int[] top = new int[]{784, 25 * 25, 10 * 10, 5 * 5, numClasses * 4, numClasses};
        builder.addOnlyFCL(top, new TanH(), true);
        //builder.addOnlyFastLayer(new int[]{784, 25 * 25, numClasses * 2, numClasses}, new TanH());


        NeuralNetwork nn = builder.getModel();

        nn = LoadModel.loadModel("fcl_emnist_letter_weights.csv");
        nn.printSummary();
        nn.test(x_test, y_test);

        double lr = 0.4;
        System.out.println("Train Data Size: " + x_train.length);
        System.out.println("Architektur: " + Arrays.toString(top));
        System.out.println("lr: " + lr);
        nn.setLoss(new MSE());
        nn.train(5, x_train, y_train, x_test, y_test, lr);
        nn.writeModelFast("fcl_emnist_letter_weights2.csv");
        nn.train(5, x_train, y_train, x_test, y_test, lr);
        nn.writeModelFast("fcl_emnist_letter_weights2.csv");

    }
}
