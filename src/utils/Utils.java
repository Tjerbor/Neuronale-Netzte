package utils;

import layers.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Arrays;
import java.util.Random;

public class Utils {


    static Random r = new Random(); //random to generate missing weights.


    /**
     * add biases to teh batch output.
     *
     * @param inputs output from calculated weights x batch_input.
     * @param bias   biases of layer
     * @return gibt die inputs plus addierten bias zurück.
     */
    public static double[][] add_biases(double[][] inputs, double[] bias) {
        double[][] out = new double[inputs.length][inputs[0].length];


        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                out[i][j] += inputs[i][j] + bias[j];
            }

        }

        return out;

    }

    /**
     * add biases to teh batch output.
     *
     * @param inputs output from calculated weights x batch_input.
     * @param bias   constant bias.
     * @return
     */
    public static double[][] add_biases(double[][] inputs, double bias) {
        double[][] out = new double[inputs.length][inputs[0].length];

        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                out[i][j] += inputs[i][j] + bias;
            }

        }

        return out;

    }

    /**
     * added bias for single computed input.
     *
     * @param inputs input -> calculated through weights x input
     * @param bias   biases of layer.
     * @return
     */
    public static double[] add_bias(double[] inputs, double[] bias) {

        for (int i = 0; i < bias.length; i++) {
            inputs[i] += bias[i];
        }
        return inputs;
    }

    public static double[] add_bias(double[] inputs, double bias) {
        for (int i = 0; i < inputs.length; i++) {
            inputs[i] += bias;
        }
        return inputs;
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

    public static double[] matmul2d_1d(double[][] wT, double[] input) {

        System.out.println(Arrays.deepToString(wT));
        System.out.println(wT.length);

        double[] out = new double[wT.length];
        System.out.println(Arrays.toString(input));
        for (int e = 0; e < wT.length; e++) {
            for (int x = 0; x < wT[0].length; x++) {
                for (int j = 0; j < 1; j++) {
                    out[e] += wT[e][x] * input[x];
                }
            }
        }
        System.out.println(Arrays.toString(out));
        return out;
    }


    /**
     * function calculates Dot-Produkt. with single Bias value.
     *
     * @param inputs -> look up
     * @param biases -> constant bias
     * @return
     */

    public static double[] dotProdukt_1D(double[][] inputs, double biases) {
        double[] output = new double[inputs[0].length];
        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                output[j] += inputs[i][j] * biases;
            }
        }

        return output;

    }

    /**
     * @param a Matrix a
     * @param b Matrix b
     * @return returns the calculated matrix with size am x bn
     */
    public static double[][] matmul2D(double[][] a, double[][] b) throws Exception {

       
        if (a[0].length != b.length) {
            throw new Exception("Mismatching Shape " + Integer.toString(a[0].length) + " " + Integer.toString(b[0].length));
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
     * Berechnet die neuen Biases werte.
     *
     * @param biases          -> Biases at the time.
     * @param output_gradient -> calculates gradient of Biases. (Sum of delta-Inputs)
     * @param learning_rate   -> paramtere decides how strong the Network learnt.
     * @return the new biases value.
     */
    public static double[] updateBiases(double[] biases, double[] output_gradient, double learning_rate) {


        for (int i = 0; i < biases.length; i++) {
            biases[i] = biases[i] - (learning_rate * output_gradient[i]);
        }


        return biases;
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
     * updates the weight of the NN.
     *
     * @param weights         actuell weights of the layer.
     * @param output_gradient -> output gradient of the layer.
     * @param learning_rate   -> paramter decides how strong the NN learns. or layer, could change
     *                        for every layer.
     * @return
     */

    public static double[][] updateWeights(double[][] weights, double[][] output_gradient, double learning_rate) {


        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[1].length; j++) {
                weights[i][j] = weights[i][j] - (learning_rate * output_gradient[i][j]);
            }

        }


        return weights;
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
     * die Biases are on the last index.
     *
     * @param a weights as input.
     * @return Biases
     */
    public static double[] split_for_biases(double[][] a) {
        return a[a.length - 1];
    }

    /**
     * takes every dimension but the last one. Because the last dim stores baises.
     *
     * @param a input weights.
     * @return weighst.
     */
    public static double[][] split_for_weights(double[][] a) {
        double[][] weights = new double[a.length - 1][a[0].length];
        for (int i = 0; i < a.length - 1; i++) {
            weights[i] = a[i];
        }
        return weights;
    }

    /**
     * Forward Pas for the 1 layer.
     *
     * @param inputs  inputs of the layer.
     * @param weights weights of the layer.
     * @param biases  biases of the layer
     * @return computed output
     * @throws Exception if matmul got a mismatching Shape.
     */
    public static double[][] doForward(double[][] inputs, double[][] weights, double[] biases) throws Exception {

        double[][] outputs;
        outputs = Utils.matmul2D(inputs, weights);
        outputs = Utils.add_biases(outputs, biases);
        return outputs;


    }

    /**
     * Forward Pas for the 1 layer.
     *
     * @param inputs  inputs of the layer.
     * @param weights weights of the layer.
     * @param biases  constant biases of the layer
     * @return computed output
     * @throws Exception if matmul got a mismatching Shape.
     */
    public static double[][] doForward(double[][] inputs, double[][] weights, double biases) throws Exception {


        double[][] outputs;
        outputs = Utils.matmul2D(inputs, weights);
        outputs = Utils.add_biases(outputs, biases);
        return outputs;


    }

    /**
     * Forward Pas for the 1 layer.
     *
     * @param input   Single input of the layer.
     * @param weights weights of the layer.
     * @param biases  constant biases of the layer
     * @return computed output
     * @throws Exception if matmul got a mismatching Shape.
     */
    public static double[] doForward(double[] input, double[][] weights, double[] biases) throws Exception {
        double[] output = new double[input.length];
        output = Utils.dotProdukt_1D(Utils.tranpose(weights), input);
        output = Utils.add_bias(output, biases);

        return output;

    }

    /**
     * Forward Pas for the 1 layer.
     *
     * @param input   Single input of the layer.
     * @param weights weights of the layer.
     * @param biases  constant biases of the layer
     * @return computed output
     * @throws Exception if matmul got a mismatching Shape.
     */
    public static double[] doForward(double[] input, double[][] weights, double biases) throws Exception {
        double[] output = new double[input.length];
        output = Utils.dotProdukt_1D(Utils.tranpose(weights), input);
        output = Utils.add_bias(output, biases);

        return output;

    }

    /**
     * creates a string of the weights and biases
     *
     * @param w weights of the layer.
     * @param b biases of the layer.
     * @return the String
     */
    public static String weightsAndBiases_toString(double[][] w, double[] b) {
        String s_out = "Biases: [";
        for (int k = 0; k < b.length; k++) {
            s_out += String.valueOf(b[k]);
        }
        s_out += "]\n";


        s_out += "weights:\n[";
        for (int i = 0; i < w.length; i++) {
            s_out += "[";
            for (int j = 0; j < w[0].length; j++) {
                s_out += String.valueOf(w[i][j]);
                if (j < w[0].length - 1) {
                    s_out += ", ";
                }

            }
            if (i < w.length - 1) {
                s_out += "],\n";
            } else {
                s_out += "]";
            }


        }
        return s_out + "]\n";
    }

    /**
     * creates a string of the weights and biases
     *
     * @param w       weights of the layer.
     * @param b       biases of the layer.
     * @param oneBias has only One bias value so do not need to print other values of layer.
     * @return the String
     */
    public static String weightsAndBiases_toString(double[][] w, double[] b, boolean oneBias) {
        String s_out;
        if (oneBias) {
            s_out = "Biases: [" + String.valueOf(b[0]) + "]\n";
        } else {
            s_out = "Biases: [";
            for (int k = 0; k < b.length; k++) {
                if (k < b.length - 1) {
                    s_out += String.valueOf(b[k]) + ", ";
                } else {
                    s_out += String.valueOf(b[k]);
                }

            }
            s_out += "]\n";
        }


        s_out += "weights:\n[";
        for (int i = 0; i < w.length; i++) {
            s_out += "[";
            for (int j = 0; j < w[0].length; j++) {
                s_out += String.valueOf(w[i][j]);
                if (j < w[0].length - 1) {
                    s_out += ", ";
                }

            }
            if (i < w.length - 1) {
                s_out += "],\n";
            } else {
                s_out += "]";
            }


        }

        return s_out + "]\n";
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
            a = new Tanh();
        } else if (name.equals("sigmoid")) {
            a = new Sigmoid();
        } else if (name.equals("softmax")) {
            a = new Softmax();
        } else if (name.equals("semi")) {
            a = new Semi_Linear();
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

    public static void printMatrix(double[][] a) {

        for (int i = 0; i < a.length; i++) {
            for (int k = 0; k < a[0].length; k++) {
                System.out.print(String.valueOf(a[i][k] + " "));
                if (k == a[0].length - 1) {
                    System.out.print("\n");
                }
            }
        }
    }

    public static double[][][] read_own_weights(String fpath) throws Exception {

        try {
            String line;
            BufferedReader br = new BufferedReader(new FileReader(fpath));

            // Condition holds true till
            // there is character in a string
            while ((line = br.readLine()) != null) {
                String[] values = line.split("\t");
                values = values[1].split(";");

            }
        } catch (Exception e) {
            System.out.println(e);
        }


        return null;
    }

    public static String weightsAndBiases_export(double[][] w, double[] b) {
        String s_out = "";
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[0].length; j++) {
                s_out += String.valueOf(w[i][j]);
                if (j < w[0].length - 1) {
                    s_out += ";";
                }
            }
            s_out += "\n";
        }
        for (int k = 0; k < b.length; k++) {
            if (k < b.length - 1) {
                s_out += String.valueOf(b[k]) + ";";
            } else {
                s_out += String.valueOf(b[k]);
            }

        }
        return s_out + "\n";
    }


}



