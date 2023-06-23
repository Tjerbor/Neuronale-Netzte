package autograd;

import autograd.nn.MLP;
import loss.MSE;
import main.MNIST;
import utils.Utils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class RunNN {


    static double acc;

    public static int argmaxTensor(List<Tensor> a) {
        int d = 0;
        double value = a.get(0).data;
        for (int i = 1; i < a.size(); i++) {
            if (a.get(i).data > value) {
                d = i;
                value = a.get(i).data;
            }
            ;
        }
        return d;
    }


    public static Tensor sumUpLosses(List<Tensor> losses) {


        Tensor data_loss = new Tensor(0);
        for (Tensor t : losses) {
            data_loss.add(t);
        }

        data_loss = data_loss.mult(new Tensor(1).div(losses.size()));

        return data_loss;


    }
    
    public static void main(String[] args) throws IOException {

        double[][][] trainingData = MNIST.read("data/mnist/train-images-idx3-ubyte.gz", "data/mnist/train-labels-idx1-ubyte.gz");
        double[][] x_train = trainingData[0];
        double[][] y_train = trainingData[1];


        double[][][] testData = MNIST.read("data/mnist/t10k-images-idx3-ubyte.gz", "data/mnist/t10k-labels-idx1-ubyte.gz");
        double[][] x_test = testData[0];
        double[][] y_test = testData[1];

        MSE loss = new MSE();

        double learning_rate = 1e-4;
        int epochs = 7;
        int step_size = x_train.length;

        List<Integer> ebenen = new ArrayList<>();

        ebenen.add(80);
        ebenen.add(40);
        ebenen.add(10);
        MLP model = new MLP(784, ebenen);


        //to validate after every Epoch.
        long st;
        for (int i = 0; i < epochs; i++) {
            st = System.currentTimeMillis();
            double[] out;
            double loss_per_step = 0;

            learning_rate -= (learning_rate * 0.05);
            for (int j = 0; j < step_size; j++) {

                model.zero_grad();

                List<Tensor> data = new ArrayList<>();
                List<Tensor> scores = new ArrayList<>();
                List<Tensor> losses = new ArrayList<>();
                for (double d : x_train[j]) {
                    data.add(new Tensor(d));
                }

                //forward
                scores = model.forward(data);

                for (int yi = 0; yi < y_train[j].length; yi++) {
                    Tensor tmpY = new Tensor(y_train[j][yi]);
                    losses.add((tmpY.neg().mult(scores.get(i)).add(1)).relu());

                }

                if (Utils.argmax(y_train[j]) == argmaxTensor(losses)) {
                    loss_per_step += 1;
                }


                //skip l2 regulation.
                Tensor totalLoss = sumUpLosses(losses);


                totalLoss.backward();
                for (Tensor p : model.getParameters()) {
                    p.data -= learning_rate * p.grad;
                }


            }
            System.out.println("acc: " + loss_per_step / x_train.length);
            long end = (System.currentTimeMillis() - st) / 1000;

            System.out.println("time: " + end);

        }

    }


}
