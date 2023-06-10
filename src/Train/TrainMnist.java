package Train;

import layer.MSE;
import main.MNIST;
import main.NeuralNetwork;
import utils.Array_utils;


public class TrainMnist {

    public static void main2(String[] args) throws Exception {


        double learning_rate = 0.004;
        int epochs = 20;
        String fpath = "./src/train_mnist.txt";
        double[][][] trainingData = MNIST.read(fpath, 60000);
        double[][] x_train = trainingData[0];
        double[][] y_train = trainingData[1];


        String fpath_test = "./src/test_mnist.txt";
        double[][][] testData = MNIST.read(fpath_test, 15000);
        double[][] x_test = testData[0];
        double[][] y_test = testData[1];

        //double[][][] x_train_bs = Mnist_reader.x_train_2_batch(x_train, 4);
        //double[][][] y_train_bs = Mnist_reader.y_train_2_batch(y_train, 4);

        //x_train = null;
        //y_train = null;

        double[][] y_true;

        NeuralNetwork nn = new NeuralNetwork();
        nn.create(new int[]{784, 14 * 14, 7 * 7, 10}, "tanh");
        nn.setLoss(new MSE());

        //nn.setLoss(new CategoricalCrossEntropy());


        double testLoss = 0;


        //to validate after every Epoch.
        for (int i = 0; i < epochs; i++) {
            //System.out.println("LearningRate: " + learning_rate);
            double[][] x_train2 = Array_utils.copyArray(x_train);
            double[][] y_train2 = Array_utils.copyArray(y_train);
            nn.train_single(1, x_train2, y_train2, learning_rate);
            learning_rate -= learning_rate / 10;
            double[][] x_test2 = Array_utils.copyArray(x_test);
            double[][] y_test2 = Array_utils.copyArray(y_test);
            testLoss = nn.test_single(x_test2, y_test2);
            if (testLoss > 0.90) {
                nn.exportWeights("weights_test" + i + "_" + ".txt");
            }


        }

        nn.test_single(x_test, y_test);
        nn.exportWeights("weights_test.txt");
        System.out.println("Wrote Weights.");
    }

    public static void main3(String[] args) throws Exception {


        double learning_rate = 0.01;
        int epochs = 50;
        String fpath = "./src/train_mnist.txt";
        double[][][] trainingData = MNIST.read(fpath, 60000);
        double[][] x_train = trainingData[0];
        double[][] y_train = trainingData[1];


        String fpath_test = "./src/test_mnist.txt";
        double[][][] testData = MNIST.read(fpath_test, 15000);
        double[][] x_test = testData[0];
        double[][] y_test = testData[1];


        double[][][] x_train_bs = MNIST.x_train_2_batch(x_train, 4);

        double[][][] y_train_bs = MNIST.y_train_2_batch(y_train, 4);

        //x_train = null;
        //y_train = null;

        double[][] y_true;

        NeuralNetwork nn = new NeuralNetwork();
        nn.create(new int[]{784, 49 * 49, 7 * 7, 10}, "tanh");
        nn.setLoss(new MSE());


        double testLoss = 0;


        //to validate after every Epoch.
        for (int i = 0; i < epochs; i++) {
            System.out.println("LearningRate: " + learning_rate);
            double[][][] x_train_bs2 = Array_utils.copyArray(x_train_bs);
            double[][][] y_train_bs2 = Array_utils.copyArray(y_train_bs);
            nn.train_with_batch(1, x_train_bs, y_train_bs, learning_rate);
            learning_rate -= learning_rate / 10;
            double[][] x_test2 = Array_utils.copyArray(x_test);
            double[][] y_test2 = Array_utils.copyArray(y_test);
            testLoss = nn.test_single(x_test2, y_test2);
            if (testLoss > 0.90) {
                nn.exportWeights("weights_test" + i + "_" + ".txt");
            }


        }

        nn.test_single(x_test, y_test);
        nn.exportWeights("weights_test_bs.txt");
        System.out.println("Wrote Weights.");
    }


    public static void main(String[] args) throws Exception {


        main2(new String[]{""});

        /**

         double learning_rate = 1e-4;
         int epochs = 10;

         String fpath = "/home/dblade/Documents/Neuronale-Netzte/src/train_mnist.txt";
         double[][] x_train = Mnist_reader.getTrainData_x(fpath);
         double[][] y_train = Mnist_reader.getTrainData_y(fpath);

         double[][][] x_train_bs = Mnist_reader.x_train_2_batch(x_train, 4);
         double[][][] y_train_bs = Mnist_reader.y_train_2_batch(y_train, 4);

         x_train = null;
         y_train = null;

         double[][] y_true;

         double loss_per_epoch;

         int step_size = x_train_bs.length;

         double[] step_losses = new double[step_size];
         Losses loss = new MSE();

         long timeStart = 0;
         for (int i = 0; i < epochs; i++) {
         timeStart = System.currentTimeMillis();
         for (int j = 0; j < step_size; j++) {
         double[][] outs;
         y_true = Array_utils.copyArray(y_train_bs[j]);
         outs = computeAll(layers, Array_utils.copyArray(x_train_bs[j]));
         step_losses[j] = loss.forward(outs, y_true);
         //calculates prime Loss
         outs = loss.backward(outs, y_true);
         // now does back propagation //updates values.
         computeAllBackward(layers, outs, learning_rate);
         System.out.println();
         }
         long timeEnd = System.currentTimeMillis();
         System.out.println(timeEnd - timeStart);
         loss_per_epoch = Utils.sumUpLoss(step_losses, step_size);
         System.out.println("Loss per epoch: " + loss_per_epoch);
         }
         **/


    }


}



