package Train;

import layer.TanH;
import main.MNIST;
import main.NeuralNetwork;

public class TrainNN {


    public static void main(String[] args) throws Exception {


        NeuralNetwork nn = new NeuralNetwork();

        double[][][] trainingData = MNIST.read("data/mnist/train-images-idx3-ubyte.gz", "data/mnist/train-labels-idx1-ubyte.gz");
        double[][] x_train = trainingData[0];
        double[][] y_train = trainingData[1];


        double[][][] testData = MNIST.read("data/mnist/t10k-images-idx3-ubyte.gz", "data/mnist/t10k-labels-idx1-ubyte.gz");
        double[][] x_test = testData[0];
        double[][] y_test = testData[1];

        int epochs = 5;
        double learning = 0.1;


        nn.create(new int[]{784, 49, 20, 10}, new TanH());
        nn.train_single(epochs, x_train, y_train, learning);


    }


}
