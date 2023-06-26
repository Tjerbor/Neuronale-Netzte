package main;

import extraLayer.FastLinearLayer;
import loss.MSE;

import java.io.IOException;

public class TrainNN {


    public static void main(String[] args) throws IOException {


        double[][][] trainingData = MNIST.read("data/mnist/train-images-idx3-ubyte.gz", "data/mnist/train-labels-idx1-ubyte.gz");
        double[][] x_train = trainingData[0];
        double[][] y_train = trainingData[1];


        double[][][] testData = MNIST.read("data/mnist/t10k-images-idx3-ubyte.gz", "data/mnist/t10k-labels-idx1-ubyte.gz");
        double[][] x_test = testData[0];
        double[][] y_test = testData[1];


        NN_New nn = new NN_New();


        FastLinearLayer f = new FastLinearLayer(784, 40, 0.4);
        FastLinearLayer f_out = new FastLinearLayer(40, 10, 0.4);

        f.setUseBiases(false);
        f.setLearningRate(0.4);
        f_out.setUseBiases(false);
        f.setLearningRate(0.4);

        LayerNew[] ls = new LayerNew[]{f, f_out};

        //nn.add(new FullyConnectedLayerNew(784, 80, new TanH()));
        //nn.add(new FullyConnectedLayerNew(80, 10, new TanH()));
        nn.setLoss(new MSE());
        //nn.setOptimizer(new AdamNew());

        nn.create(ls);
        nn.build();

        System.out.println("Started Training");
        nn.trainNew(10, x_train, y_train, x_test, y_test, 0.4);

    }
}
