package examples;

import builder.NetworkBuilder;
import function.TanH;
import loss.MSE;
import main.MNIST;
import main.NeuralNetwork;
import utils.TrainUtils;

import java.io.IOException;

public class ExpTrainBatch {


    public static void main(String[] args) throws IOException {


        int numClasses = 26;
        double[][][] trainingData = MNIST.read("data/emnist/emnist-letters-train-images-idx3-ubyte.gz", "data/emnist/emnist-letters-train-labels-idx1-ubyte.gz", numClasses);
        double[][] x_train = trainingData[0];
        double[][] y_train = trainingData[1];


        double[][][] testData = MNIST.read("data/emnist/emnist-letters-test-images-idx3-ubyte.gz", "data/emnist/emnist-letters-test-labels-idx1-ubyte.gz", numClasses);
        double[][] x_test = testData[0];
        double[][] y_test = testData[1];

        NetworkBuilder builder = new NetworkBuilder();
        builder.addOnlyFCL(new int[]{784, 20 * 20, numClasses * 2, numClasses}, new TanH(), true);
        //builder.addOnlyFastLayer(new int[]{784, 25 * 25, numClasses * 2, numClasses}, new TanH());

        NeuralNetwork nn = builder.getModel();

        int batchSize = 4;

        double[][][] x_trainB = TrainUtils.getBatch(x_train, batchSize);
        double[][][] y_trainB = TrainUtils.getBatch(y_train, batchSize);

        double[][][] x_testB = TrainUtils.getBatch(x_test, batchSize);
        double[][][] y_testB = TrainUtils.getBatch(y_test, batchSize);

        System.out.println(x_train.length + " : " + y_train.length);
        nn.setLoss(new MSE());
        nn.trainBatch(10, x_trainB, y_trainB, x_testB, y_testB, 0.1);
        nn.writeModel("nn_emnist_letter_weights.txt");


    }
}
