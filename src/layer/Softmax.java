package layer;

import utils.Array_utils;
import utils.Utils;

public class Softmax extends Activation {

    static double[][] outputs = new double[][]{{0.09003057, 0.24472847, 0.66524096},
            {0.09003057, 0.24472847, 0.66524096},
            {0.09003057, 0.24472847, 0.66524096}};
    double exp = 0;
    double[][] outputsForward;

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

        int l = outputs[0].length;
        double[][] singleOutputsTmpT = new double[1][l];
        double[][] singleOutputsTmp;
        double[][] tmp2;


        for (int i = 0; i < dvalues.length; i++) {

            tmp = diagflat(outputs[i]);

            singleOutputsTmp = reshapeMinusOne(outputs[i]);
            singleOutputsTmpT[0] = outputs[i]; //Transposed
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

        outputsForward = inputs.clone();
        return inputs;

    }
}
