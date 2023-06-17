package layer;

import utils.Array_utils;
import utils.Utils;

public class CategoricalCrossEntropy extends Losses {

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


    public double forward(double[] y_pred, double[] y_true) {

        int sum = 0;

        double sanityValue = Math.pow(10, -100);

        int i = Utils.argmax(y_true);

        sum += y_true[i] * Math.log(y_pred[i] + sanityValue);
        return -sum;
    }

    public double forward(double[][] y_pred, double[][] y_true) {

        int batch_size = y_true.length;
        clip(y_pred, 1e-7, 1 - 1e-7);

        double output = 0;

        double[][] c = new double[y_true.length][y_true[0].length];
        for (int i = 0; i < y_true.length; i++) {
            for (int j = 0; j < y_true[0].length; j++) {
                c[i][j] = y_true[i][j] * y_pred[i][j];
            }
        }


        double[] corrrect_confident = Array_utils.sum_axis_1(c);

        double neg_log = 0;

        for (int i = 0; i < corrrect_confident.length; i++) {
            neg_log += -Math.log(corrrect_confident[i]);
        }


        return neg_log / batch_size;
    }

    public double[][] backward(double[][] y_pred, double[][] y_true) {

        double[][] output = new double[y_true.length][y_true[0].length];
        for (int i = 0; i < y_true.length; i++) {
            for (int j = 0; j < y_true[0].length; j++) {
                output[i][j] = (-y_true[i][j] / y_pred[i][j]) / y_true.length; //divide by batch size.
            }
        }
        return output;
    }

    public double[] backward(double[] y_pred, double[] y_true) {

        double[] c = new double[y_true.length];
        double sanityValue = Math.pow(10, -100);

        int i = Utils.argmax(y_true);

        c[i] = -(y_true[i] / y_pred[i] + sanityValue);
        return c;

    }


}