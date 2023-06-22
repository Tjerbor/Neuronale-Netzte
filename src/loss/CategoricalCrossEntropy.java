package loss;

import utils.Array_utils;
import utils.Utils;

public class CategoricalCrossEntropy implements Loss {
    public static void clip(double[] a, double min, double max) {

        for (int i = 0; i < a.length; i++) {
            if (a[i] > max) {
                a[i] = max;
            } else if (a[i] < min) {
                a[i] = min;
            }
        }
    }

    public static void clip(double[][] a, double min, double max) {

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                if (a[i][j] > max) {
                    a[i][j] = max;
                } else if (a[i][j] < min) {
                    a[i][j] = min;
                }
            }

        }


    }

    public double maxVal(double[] a) {

        double d = 1 - 1e-7;
        for (int i = 0; i < a.length; i++) {
            if (a[i] > d && a[i] != 0) {
                d = a[i];
            }
        }

        return d;

    }

    @Override
    public double forward(double[] actual, double[] expected) {

        int sum = 0;

        double sanityValue = Math.pow(10, -100);

        int i = Utils.argmax(expected);

        sum += expected[i] * Math.log(actual[i] + sanityValue);
        return -sum;
    }

    @Override
    public double forward(double[][] actual, double[][] expected) {

        int batch_size = expected.length;
        clip(actual, 1e-7, 1 - 1e-7);

        double output = 0;

        double[][] c = new double[expected.length][expected[0].length];
        for (int i = 0; i < expected.length; i++) {
            for (int j = 0; j < expected[0].length; j++) {
                c[i][j] = expected[i][j] * actual[i][j];
            }
        }


        double[] corrrect_confident = Array_utils.sum_axis_1(c);

        double neg_log = 0;

        for (int i = 0; i < corrrect_confident.length; i++) {
            neg_log += -Math.log(corrrect_confident[i]);
        }


        return neg_log / batch_size;
    }

    @Override
    public double[] backward(double[] actual, double[] expected) {

        double[] c = new double[expected.length];
        double sanityValue = Math.pow(10, -100);

        int i = Utils.argmax(expected);

        c[i] = -(expected[i] / actual[i] + sanityValue);
        return c;

    }

    @Override
    public double[][] backward(double[][] actual, double[][] expected) {

        double[][] output = new double[expected.length][expected[0].length];
        for (int i = 0; i < expected.length; i++) {
            for (int j = 0; j < expected[0].length; j++) {
                output[i][j] = (-expected[i][j] / actual[i][j]) / expected.length; //divide by batch size.
            }
        }
        return output;
    }
}
