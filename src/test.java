import layers.*;

import java.util.Arrays;

public class test {


    // ungenauigkeit numpy 108.94117755: Java 107.94117736599989
    public static void printArr(double[][] a) {

        for (int i = 0; i < a.length; i++) {
            for (int k = 0; k < a[0].length; k++) {
                System.out.print(String.valueOf(a[i][k] + " "));
                if (k == a[0].length - 1) {
                    System.out.print("\n");
                }
            }
        }
    }

    public static void main(String[] args) throws Exception {

        FullyConnectedLayer d1 = new FullyConnectedLayer(784, 20, true);
        FullyConnectedLayer d2 = new FullyConnectedLayer(20, 10, true);
        Activation act = new Tanh_v2();
        Losses loss = new MSE();


        String fpath = "src/utils/mnist_data_full.txt";
        double[][] y_train = Mnist_reader.getTrainData_y(fpath);
        double[][] x_train = Mnist_reader.getTrainData_x(fpath);

        double[][][] x_train2 = Mnist_reader.x_train_2_batch(x_train, 4);
        double[][][] y_train2 = Mnist_reader.x_train_2_batch(y_train, 4);


        if (x_train2[0] == x_train2[5] || x_train2[0] == x_train2[4]) {
            System.out.println("Error");
            throw new RuntimeException();
        }

        Layer[] l = new Layer[4];
        l[0] = new FullyConnectedLayer(784, 20);
        l[0].setWeights(784, 20);
        l[1] = new Tanh();
        l[2] = new FullyConnectedLayer(20, 10);
        l[2].setWeights(20, 10);
        l[3] = new Tanh();

        NeuralNetwork nn = new NeuralNetwork();
        nn.create(l);
        nn.loss = new MSE();
        System.out.println(Arrays.toString(nn.topologie));
        nn.train_with_batch(5, x_train2, y_train2, 0.5);

        
    }
}
