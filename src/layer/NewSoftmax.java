package layer;

import utils.Array_utils;

public class NewSoftmax {


    double[] outputForward;
    double[][][] input3D;
    double[][][][] inputs3D;


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
