package Train;

import extraLayer.FastLinearLayer;
import loss.MSE;
import main.LayerNew;
import main.MNIST;
import main.NeuralNetwork;

import java.io.IOException;

public class TrainNN_Stats {


    public static void main(String[] args) throws IOException {


        double[][][] trainingData = MNIST.read("data/mnist/train-images-idx3-ubyte.gz", "data/mnist/train-labels-idx1-ubyte.gz");
        double[][] x_train = trainingData[0];
        double[][] y_train = trainingData[1];


        double[][][] testData = MNIST.read("data/mnist/t10k-images-idx3-ubyte.gz", "data/mnist/t10k-labels-idx1-ubyte.gz");
        double[][] x_test = testData[0];
        double[][] y_test = testData[1];


        NeuralNetwork nn = new NeuralNetwork();


        FastLinearLayer f = new FastLinearLayer(784, 40, 0.1);
        FastLinearLayer f_out = new FastLinearLayer(40, 10, 0.1);

        f.setUseBiases(false);
        f.setLearningRate(0.4);
        f_out.setUseBiases(false);
        f.setLearningRate(0.4);

        String s = "";
        LayerNew[] ls = new LayerNew[]{f, f_out};

        //nn.add(new FullyConnectedLayerNew(784, 80, new TanH()));
        //nn.add(new FullyConnectedLayerNew(80, 10, new TanH()));
        nn.setLoss(new MSE());
        //nn.setOptimizer(new AdamNew());

        nn.create(ls);
        nn.build();

        System.out.println("Started Training");
        s += nn.trainTesting(10, x_train, y_train, x_test, y_test, 0.001);
        s += nn.trainTesting(10, x_train, y_train, x_test, y_test, 0.01);
        s += nn.trainTesting(10, x_train, y_train, x_test, y_test, 0.1);
        s += nn.trainTesting(10, x_train, y_train, x_test, y_test, 0.2);
        s += nn.trainTesting(10, x_train, y_train, x_test, y_test, 0.3);
        s += nn.trainTesting(10, x_train, y_train, x_test, y_test, 0.4);

        System.out.println(s);
    }
}
