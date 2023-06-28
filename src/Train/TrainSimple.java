package Train;

import builder.NetworkBuilder;
import layer.FastLinearLayer;
import layer.Layer;
import loss.MSE;
import main.MNIST;
import main.NeuralNetwork;

import java.io.IOException;

public class TrainSimple {


    public static void main(String[] args) throws IOException {

        double[][][] trainingData = MNIST.read("data/mnist/train-images-idx3-ubyte.gz", "data/mnist/train-labels-idx1-ubyte.gz");
        double[][] x_train = trainingData[0];
        double[][] y_train = trainingData[1];


        double[][][] testData = MNIST.read("data/mnist/t10k-images-idx3-ubyte.gz", "data/mnist/t10k-labels-idx1-ubyte.gz");
        double[][] x_test = testData[0];
        double[][] y_test = testData[1];

        NeuralNetwork nn = new NeuralNetwork();

        FastLinearLayer f = new FastLinearLayer(784, 40);
        FastLinearLayer f_out = new FastLinearLayer(40, 10);

        f.setUseBiases(true);
        f.setLearningRate(0.4);
        f_out.setUseBiases(true);
        f.setLearningRate(0.4);

        String s = "";
        Layer[] ls = new Layer[]{f, f_out};


        nn.create(ls);

        NetworkBuilder builder = new NetworkBuilder(784);
        builder.addFastLayer(80);
        builder.addFastLayer(40);
        builder.addFastLayer(10);

        nn = builder.getModel();


        nn.setLoss(new MSE());
        nn.train(5, x_train, y_train, x_test, y_test, 0.3);
        nn.writeModel("nn_weights.txt");
        //nn.test(x_test, y_test);


        String dirFpath2 = "data/selfmade_data";
        testData = LoadOwn.getImages(dirFpath2);

        double[][] y_testOwn = testData[1];
        double[][] x_testOwn = testData[0];

        double[][] x_testOwn2 = new double[x_testOwn.length][x_testOwn[0].length];
        for (int i = 0; i < x_testOwn.length; i++) {
            for (int j = 0; j < x_testOwn[0].length; j++) {
                x_testOwn2[i][783 - j] = x_testOwn[i][j];
            }
        }


        nn.printTestStats(x_testOwn2, y_testOwn);


    }
}
