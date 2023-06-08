package Train;

import layer.FullyConnectedLayer;
import layer.Layer;
import layer.MSE;
import layer.TanH;
import main.Mnist_reader;
import main.NeuralNetwork;


public class TrainMnist {

    public static void computeAllBackward(Layer[] layers, double[][] dinputs, double learning_rate) {


        double[][] doutputs = dinputs;
        for (int i = 0; i < layers.length; i++) {
            doutputs = layers[layers.length - 1 - i].backward(doutputs, learning_rate);
        }

    }

    public static double[][] computeAll(Layer[] layers, double[][] inputs) {

        double[][] outputs = inputs;
        for (Layer layer : layers) {
            outputs = layer.forward(outputs);

        }
        return outputs;
    }


    public static void main2(String[] args) throws Exception {


        FullyConnectedLayer f1 = new FullyConnectedLayer(784, 7 * 7);
        TanH act = new TanH();
        FullyConnectedLayer f3 = new FullyConnectedLayer(7 * 7, 10);
        //SoftmaxCategoricalCrossEntropy act_out_loss = new SoftmaxCategoricalCrossEntropy();


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

        Layer[] layers = new Layer[]{f1, act, f3, act};
        NeuralNetwork nn = new NeuralNetwork();
        nn.create(new int[]{784, 7 * 7, 10}, "tanh");

        nn.setLoss(new MSE());

        nn.train_with_batch(epochs, x_train_bs, y_train_bs, learning_rate);

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



