package Train;

import layer.MSE;
import main.MNIST;
import main.NeuralNetwork;

public class TrainApi {


    public static void main(String[] args) throws Exception {


        NeuralNetwork nn = new NeuralNetwork();

        nn.create(new int[]{784, 49, 20, 10}, new String[]{"tanh", "tanh", "tanh"});


        String fpath = "./src/train_mnist.txt";
        double[][][] trainingData = MNIST.read(fpath, 60000);
        double[][] x_train = trainingData[0];
        double[][] y_train = trainingData[1];

        String fpath_test = "./src/test_mnist.txt";
        double[][][] testData = MNIST.read(fpath_test, 15000);
        double[][] x_test = testData[0];
        double[][] y_test = testData[1];


        nn.setLoss(new MSE());

        System.out.println(nn);

        nn.train_single(20, x_train, y_train, 0.4);
        nn.test_single(x_test, y_test);
        nn.exportWeights("weights_train.txt");

    }

}
