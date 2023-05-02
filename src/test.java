import layers.*;

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

        //System.out.println(Arrays.toString(x_train2[0][0]));
        //System.out.println(Arrays.toString(x_train2[0][1]));
     /*
        String strStep = "";
        for (int stepS = 0; stepS < x_train2.length; stepS++) {

            strStep = String.valueOf(stepS) + ": ";
            double[][] out = d1.forward(x_train2[stepS]);
            System.out.println(strStep + "out1: ");
            printArr(out);

            out = act.forward(out);
            System.out.println(strStep + "act1: ");
            printArr(out);

            out = d2.forward(out);
            System.out.println(strStep + "out2: ");
            printArr(out);

            out = act.forward(out);
            System.out.println(strStep + "act2: ");
            printArr(out);

            double loss_val = loss.forward(out, y_train2[stepS]);
            System.out.print(strStep + "Loss out: ");
            System.out.println(loss_val);

            //backward
            double[][] grad = loss.backward(out, y_train2[stepS]);
            System.out.print(strStep + "Loss backward: ");
            printArr(grad);

            grad = act.backward(grad);
            System.out.print(strStep + ": backward 1: ");
            printArr(grad);

            grad = d2.backward(grad, 0.5);
            System.out.print(strStep + "backward 2: ");
            printArr(grad);

            grad = act.backward(grad);
            System.out.print(strStep + "backward 1: ");
            printArr(grad);

            grad = d1.backward(grad, 0.5);
            System.out.print(strStep + "backward 2: ");
            printArr(grad);

            System.out.print(strStep + "weights d2: ");
            printArr(d2.weights);

            if (stepS == 0) {

                System.out.print(strStep + "Loss out: ");
                System.out.println(loss_val);
                break;
            }

        }
        */


    }
}
