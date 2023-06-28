package examples;

import builder.NetworkBuilder;
import loss.Loss;
import loss.MSE;
import main.MNIST;
import main.NeuralNetwork;
import utils.Array_utils;
import utils.Matrix;
import utils.TrainUtils;
import utils.Utils;

import java.io.IOException;
import java.util.List;

public class ExpTrain {

    public static void main(String[] args) throws IOException {


        int numFilter = 8;
        int numClasses = 10;
        int[] inputShape = new int[]{28, 28, 1};
        int kernelSize = 5;
        int strides = 2; //stepSize.
        NetworkBuilder builder = new NetworkBuilder(inputShape);
        builder.addConv2D_Last(numFilter, kernelSize, strides);
        builder.addDropout(0.5);
        builder.addMaxPooling2D_Last(); //uses for standard strides2 and poolSize2
        builder.addBatchNorm();
        builder.addFlatten();
        builder.addFastLayer(numClasses);

        NeuralNetwork nn = builder.getModel();


        double[][][] trainingData = MNIST.read("data/mnist/train-images-idx3-ubyte.gz", "data/mnist/train-labels-idx1-ubyte.gz");

        double[][][] testData = MNIST.read("data/mnist/t10k-images-idx3-ubyte.gz", "data/mnist/t10k-labels-idx1-ubyte.gz");


        int batchSize = 4;
        double learningRate = 0.4;
        int epochs = 5;
        long st = 0;
        long end;
        Loss loss = new MSE();
        int step_size = trainingData[0].length / batchSize;

        List tmp;
        nn.getLastLayer().setLearningRate(learningRate);
        for (int i = 0; i < epochs; i++) {
            st = System.currentTimeMillis();
            double[][] out;
            double[] stepLosses = new double[step_size];

            for (int j = 0; j < step_size; j++) {
                tmp = TrainUtils.yieldXY_Last(batchSize, trainingData[0], trainingData[1], j);
                double[][][][] x_train = (double[][][][]) tmp.get(0);
                double[][] y_train = (double[][]) tmp.get(1);

                nn.getFristLayer().forward(new Matrix(Array_utils.copyArray(x_train)));
                //calculates Loss
                Matrix out2 = nn.getLastLayer().getOutput();
                System.out.println("print Data: " + out2.getData());
                out = out2.getData2D();
                stepLosses[j] = loss.forward(out, y_train);
                //calculates backward Loss
                out = loss.backward(out, y_train);
                // now does back propagation
                nn.getLastLayer().setIterationAt(i);
                nn.getLastLayer().backward(new Matrix(out));
            }

            end = System.currentTimeMillis();
            System.out.println("Time: " + ((end - st) / 1000));
            System.out.println("Loss: " + Utils.mean(stepLosses));
        }


    }

}
