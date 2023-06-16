package layer;

import utils.Array_utils;
import utils.Utils;

public class NewSoftmax {


    double[] outputForward;
    double[][] outputsForward;
    double[][][] input3D;
    double[][][][] inputs3D;


    public static double[][] diagflat(double[] a) {

        double[][] out = new double[a.length][a.length];

        //Arrays.fill(out, 0);
        for (int i = 0; i < a.length; i++) {
            out[i][i] = a[i];
        }

        return out;
    }

    public static double[][] reshapeMinusOne(double[] a) {
        double[][] c = new double[a.length][1];

        for (int j = 0; j < a.length; j++) {
            c[j][0] = a[j];
        }

        return c;
    }

    public double[][] backward(double[][] dvalues) {

        double[][] out = new double[dvalues.length][dvalues[0].length];

        double[][] tmp;

        int l = outputsForward[0].length;
        double[][] singleOutputsTmpT = new double[1][l];
        double[][] singleOutputsTmp;
        double[][] tmp2;


        for (int i = 0; i < dvalues.length; i++) {

            tmp = diagflat(outputsForward[i]);

            singleOutputsTmp = reshapeMinusOne(outputsForward[i]);
            singleOutputsTmpT[0] = outputsForward[i]; //Transposed
            tmp2 = Utils.matmul2D(singleOutputsTmp, singleOutputsTmpT);


            Array_utils.subMatrix(tmp, tmp2);
            out[i] = Utils.dotProdukt_1D(tmp, dvalues[i]);

        }

        return out;


    }

    public double softmax(double x, double sum) {
        return Math.exp(x) / sum;

    }

    public double[][] forward(double[][] inputs) {

        double exp = 0;
        double sum = 0;
        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                exp += this.softmax(inputs[i][j], 1);

            }

        }
        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                inputs[i][j] += this.softmax(inputs[i][j], exp);
            }

        }

        outputsForward = Array_utils.copyArray(inputs);
        return inputs;

    }

    public double[] forward(double[] input) {

        outputForward = new double[input.length];
        double sumUp = 0;
        double[] tmp = new double[input.length];
        double[] out = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            tmp[i] = Math.exp(input[i]);
            sumUp += tmp[i];
        }
        for (int i = 0; i < input.length; i++) {

            out[i] = tmp[i] / sumUp;
        }

        outputForward = Array_utils.copyArray(out);
        return out;

    }


    public double[][] identity(int n) {


        double[][] id = new double[n][n];


        for (int i = 0; i < n; i++) {
            id[i][i] = 1;
        }

        return id;

    }

    public double[] backward(double[] grad) {


        int n = grad.length;

        double[][] id = identity(n);

        double[] out = new double[n];
        for (int i = 0; i < id.length; i++) {
            for (int j = 0; j < id.length; j++) {
                id[i][j] = ((id[i][j] - outputForward[i]) * outputForward[i]);
                out[i] += id[i][j] * grad[j];
            }


        }

        return grad;
    }


}
