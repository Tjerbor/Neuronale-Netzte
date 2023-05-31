package layer;

import utils.Array_utils;

public class CategoricalCrossEntropy {

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

    public static double forward(double[][] y_true, double[][] y_pred) {

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
            neg_log += Math.log(corrrect_confident[i]);
        }


        return neg_log / batch_size;
    }

    public static double[][] backward(double[][] y_true, double[][] y_pred) {

        double[][] output = new double[y_true.length][y_true[0].length];
        for (int i = 0; i < y_true.length; i++) {
            for (int j = 0; j < y_true[0].length; j++) {
                output[i][j] = (-y_true[i][j] / y_pred[i][j]) / y_true.length; //divide by batch size.
            }
        }
        return output;
    }

    public static double[] backward(double[] y_true, double[] y_pred) {

        double[] output = new double[y_true.length];
        for (int i = 0; i < y_true.length; i++) {
            output[i] = (-y_true[i] / y_pred[i]) / y_true.length;

        }
        return output;
    }


}