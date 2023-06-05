package utils;

import autograd.Tensor;
import layer.*;

import java.util.Arrays;
import java.util.Random;

public class Utils {
    static Random r = new Random(); //random to generate missing weights.


    public static double[] mean_axis_1(double[][] a) {

        double[] out = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                out[j] = a[i][j];

            }
        }

        for (int i = 0; i < out.length; i++) {
            out[i] /= a[0].length;
        }

        return out;

    }

    public static double[] mean_axis_0(double[][] a) {

        double[] out = new double[a[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                out[j] += a[i][j];

            }
        }

        for (int i = 0; i < out.length; i++) {
            out[i] /= a.length;
        }

        return out;

    }

    /**
     * Tranpose a given 2D array.
     *
     * @param a matrix
     * @return a.T
     */
    public static double[][] tranpose(double[][] a) {

        double[][] c = new double[a[0].length][a.length];


        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                c[j][i] = a[i][j];
            }
        }


        return c;
    }


    public static void cal_matrix_mult_scalar(double[][] a, double scalar) {

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                a[i][j] *= scalar;
            }
        }


    }

    public static void cal_matrix_minus_scalar(double[][] a, double scalar) {

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                a[i][j] -= scalar;
            }
        }


    }

    public static void addMatrix(double[][] a, double[] b) {

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                a[i][j] += b[j];
            }
        }
    }

    public static void addMatrix(double[][] a, double[][] b) {

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                a[i][j] += b[i][j];
            }
        }
    }

    /**
     * the first array will be overwritten.
     */
    public static void addMatrix(double[] a, double[] b) {
        for (int i = 0; i < a.length; i++) {
            a[i] += b[i];
        }

    }

    /**
     * calculates the new momentum Biases to update the weights.
     *
     * @param momentumBiases
     * @param dBiases
     * @param learningRate
     * @param momentum
     */
    public static void cal_baisesMomentum(double[] momentumBiases, double[] dBiases, double learningRate, double momentum) {

        for (int i = 0; i < dBiases.length; i++) {
            momentumBiases[i] = momentum * momentumBiases[i] - dBiases[i] * learningRate;
        }

    }


    public static void cal_momentumW_minus_dweights(double[][] MomentumW, double[][] dweights, double learningRate) {

        Utils.cal_matrix_mult_scalar(dweights, learningRate);

        for (int i = 0; i < MomentumW.length; i++) {
            for (int j = 0; j < MomentumW[0].length; j++) {
                MomentumW[i][j] -= dweights[i][j];
            }
        }
    }


    /**
     * adds biases to the output of the NN.
     * used for forward-Pass
     *
     * @param inputs -> use Case: weights multiplited by input-Data.
     * @param biases -> Biases of Layer.
     * @return
     */
    public static double[] dotProdukt_1D(double[][] inputs, double[] biases) {


        double[] output = new double[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                output[i] += inputs[i][j] * biases[j];
            }
        }

        return output;

    }

    public static double[][][] matmul3D(double[][][] a, double[][][] b) {


        if (a[0].length != b.length && !(b.length == 1)) {
            throw new ArithmeticException("Mismatching Shape " + (a[0].length) + " " + (b.length));
        }

        double[][][] c = new double[a.length][a[0].length][a[0][0].length];

        for (int i = 0; i < a.length; i++) {
            c[i] = matmul2D(a[i], b[i]);
        }
        return c;
    }


    public static double[][] matmul2D(double[][] a, Tensor[][] b) {

        if (a[0].length != b.length) {
            throw new ArithmeticException("Mismatching Shape " + Integer.toString(a[0].length) + " " + Integer.toString(b.length));
        }

        double[][] c = new double[a.length][b[1].length];

        for (int e = 0; e < a.length; e++) {
            for (int x = 0; x < b[1].length; x++) {
                for (int j = 0; j < a[0].length; j++) {
                    c[e][x] += a[e][j] * b[j][x].data;

                }
            }
        }


        return c;
    }

    /**
     * @param a Matrix a
     * @param b Matrix b
     * @return returns the calculated matrix with size am x bn
     */
    public static double[][] matmul2D(double[][] a, double[][] b) {

        if (a[0].length != b.length) {
            throw new ArithmeticException("Mismatching Shape " + Integer.toString(a[0].length) + " " + Integer.toString(b.length));
        }

        double[][] c = new double[a.length][b[1].length];

        for (int e = 0; e < a.length; e++) {
            for (int x = 0; x < b[1].length; x++) {
                for (int j = 0; j < a[0].length; j++) {
                    c[e][x] += a[e][j] * b[j][x];

                }
            }
        }


        return c;
    }

    /**
     * calculates 2 1D-Arrays with each other.
     * normal matrix-multiplikation
     *
     * @param input1 first Matrix
     * @param input2 second Matrix
     * @return output 2D Matrix. mx1 x 1xn
     * throws no exception. It is expected that the user knows what he is doing.
     * another Name could be Calculate1D_to_2D.
     */
    public static double[][] calcWeightGradient(double[] input1, double[] input2) {


        double[][] c = new double[input1.length][input2.length];

        int aS = input1.length;
        int bS = input2.length;

        for (int e = 0; e < aS; e++) {
            for (int x = 0; x < bS; x++) {
                //ist constant 1 weil die shape mx1 und 1xn erwartet wird.
                for (int j = 0; j < 1; j++) {
                    c[e][x] += input1[e] * input2[x];

                }
            }
        }


        return c;


    }

    /**
     * needed for Backpropagation to update the biases.
     * sum up delat values.
     *
     * @param dvalues -> delta values input for layer.
     * @return the sum.
     */
    public static double[] sumBiases(double[][] dvalues) {

        double[] a = new double[dvalues[1].length];


        for (int i = 0; i < dvalues[0].length; i++) {
            for (int j = 0; j < dvalues.length; j++) {
                a[i] += dvalues[j][i];
            }
        }

        return a;
    }

    /**
     * cerates random weights in range -1 to 1
     *
     * @param size1 m size of Matrix
     * @param size2 n size of matrix
     * @return nxm weights Matrix.
     */
    public static double[][] genRandomWeights(int size1, int size2) {

        double[][] c = new double[size1][size2];

        for (int i = 0; i < size1; i++) {
            for (int j = 0; j < size2; j++) {

                c[i][j] = genRandomWeight();


            }
        }


        return c;
    }

    public static double[][][] genRandomWeights(int size1, int size2, int size3) {

        double[][][] c = new double[size1][size2][size3];

        for (int i = 0; i < size1; i++) {
            for (int j = 0; j < size2; j++) {
                for (int k = 0; k < size3; k++) {
                    c[i][j][k] = genRandomWeight();
                }
            }
        }


        return c;
    }

    public static double sumUpLoss(double[][] losses) {
        double l_out = 0;

        for (int i = 0; i < losses.length; i++) {
            for (int j = 0; j < losses[0].length; j++) {
                l_out += losses[i][j];
            }
        }

        return l_out;
    }

    public static Activation getActivation() {
        Activation a = new Activation();
        ;
        return a;
    }

    public static Activation getActivation(String name) {
        name = name.toLowerCase();

        Activation a;
        if (name.equals("relu")) {
            a = new ReLu();
        } else if (name.equals("tanh")) {
            a = new TanH();
        } else if (name.equals("sigmoid")) {
            a = new Sigmoid();
        } else if (name.equals("softmax")) {
            a = new Softmax();
        } else if (name.equals("semi")) {
            a = new SemiLinear();
        } else {
            a = new Activation();
        }

        return a;


    }

    /**
     * generates Single weight in range -1 to 1.
     *
     * @return the random number.
     */
    public static double genRandomWeight() {
        return r.nextDouble(-1, 1);
    }

    public static double mean(double[] a) {
        int s = a.length;
        double sum = 0;

        for (int i = 0; i < s; i++) {
            sum += a[i];
        }


        return sum / s;


    }

    public static double[] meanForArray(double[] a) {
        int s = a.length;
        double sum = 0;
        double[] out = new double[s];

        for (int i = 0; i < s; i++) {
            sum += a[i];
        }

        for (int i = 0; i < s; i++) {
            out[i] = a[i] / sum;
        }

        return out;


    }

    public static double meanForArray(double[][] a) {
        int s = a.length;
        int s1 = a[0].length;
        double sum = 0;


        double[] out = new double[s];

        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s1; j++) {
                sum += a[i][j];
            }
        }


        return sum / s;


    }

    public static double[] power(double[] y_true, double[] y_pred, double factor) {
        int s = y_true.length;


        double[] calc = new double[s];
        for (int i = 0; i < s; i++) {
            calc[i] = Math.pow(y_true[i] - y_pred[i], factor);
        }


        return calc;


    }

    /**
     * Returns the indidcie of the higehst value in an array.
     *
     * @param a
     * @return
     */
    public static int argmax(double[] a) {
        int s = a.length;
        int d = -1;
        for (int i = 0; i < s; i++) {
            if (a[i] > d) {
                d = i;
            }
            ;
        }
        return d;
    }

    public static int[] argmax(double[][] a) {

        int s = a.length;
        int s1 = a[1].length;
        int[] d = new int[s];
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s1; j++) {
                if (a[i][j] > d[i]) {
                    d[i] = j;
                }
                ;
            }
        }
        return d;
    }

    public static double[][] power(double[][] y_true, double[][] y_pred, double factor) {
        int s = y_true.length;
        int s1 = y_true[0].length;


        double[][] calc = new double[s][s1];
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s1; j++) {
                calc[i][j] = Math.pow(y_true[i][j] - y_pred[i][j], factor);
            }

        }


        return calc;


    }

    public static double sumUpLoss(double[] step_losses, double step_size) {
        double sum = 0;
        int s = step_losses.length;

        for (int i = 0; i < s; i++) {
            sum += step_losses[i];
        }


        return sum / (step_size);
    }

    public static double[][] clean_inputs(double[][] inputs, int right_size) {
        int batch_size = inputs.length;

        double[][] out = new double[batch_size][right_size];
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < right_size; j++) {
                out[i][j] += inputs[i][j];
            }
        }

        return out;
    }

    public static double[] clean_input(double[] inputs, int right_size) {


        double[] out = new double[right_size];
        for (int i = 0; i < right_size; i++) {
            out[i] += inputs[i];
        }

        return out;
    }

    public static double[] getOnesBiases(int size) {
        double[] b = new double[size];
        Arrays.fill(b, 1);

        return b;
    }

    public static double[] getZeroBiases(int size) {
        double[] b = new double[size];
        Arrays.fill(b, 0);

        return b;
    }

    public static double[][] getOnesWeights(int n_input, int n_neurons) {
        double[][] w = new double[n_input][n_neurons];

        for (int i = 0; i < n_input; i++) {
            for (int j = 0; j < n_neurons; j++) {
                w[i][j] = 1;
            }
        }


        return w;
    }
}
