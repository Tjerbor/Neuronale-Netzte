package utils;

import java.text.DecimalFormat;
import java.util.Arrays;

/**
 * this class helps with array operations like print or
 * linspace to help simulate thinks.
 */
public class Array_utils {


    public static double roundDec(double value, int size) {

        String strFormat = "#.";
        for (int i = 0; i < size; i++) {
            strFormat += "#";
        }

        DecimalFormat df = new DecimalFormat(strFormat);
        String formate = df.format(value);
        //System.out.println(formate);
        formate = formate.replace(",", ".");
        double d = Double.parseDouble(formate);
        //System.out.println(d);
        return d;
    }

    public static double[][] getLinspaceWeights_wo_endpoint(int n_inputs, int n_neurons, double start, double end, int dec_precission) {

        double[][] w = new double[n_inputs][n_neurons];
        for (int i = 0; i < n_inputs; i++) {
            w[i] = linspace_wo_endpoint(start, end, n_neurons, dec_precission);

        }
        return w;


    }

    public static double[] linspace(double start, double end, int size) {

        double[] out = new double[size];
        double range = end - start;
        double step_val = range / (size - 1);


        for (int i = 0; i < size; i++) {
            out[i] = start + (step_val * i);
        }


        return out;


    }

    public static double[] linspace(double start, double end, int size, int round_size) {

        double[] out = new double[size];
        double range = end - start;
        double step_val = range / (size - 1);


        double d;
        for (int i = 0; i < size; i++) {

            try {
                d = start + (step_val * i);
                d = Array_utils.roundDec(d, round_size);
                out[i] = d;
            } catch (Exception e) {
                System.out.println(e);
                out[i] = start + (step_val * i);
            }


        }

        return out;


    }


    /**
     * Linspace without endpoint
     *
     * @param start
     * @param end
     * @param size
     * @return
     */
    public static double[] linspace_wo_endpoint(double start, double end, int size) {

        double[] out = new double[size];
        double range = end - start;
        double step_val = range / size;

        for (int i = 0; i < size; i++) {
            out[i] = start + (step_val * i);
        }

        return out;


    }

    public static double[] linspace_wo_endpoint(double start, double end, int size, int round_size) {

        double[] out = new double[size];
        double range = end - start;
        double step_val = range / (size);


        double d;
        for (int i = 0; i < size; i++) {
            d = start + (step_val * i);
            out[i] = Array_utils.roundDec(d, round_size);

        }
        return out;
    }

    /**
     * Linspace without start Point
     *
     * @param start
     * @param end
     * @param size
     * @return
     */
    public static double[] linspace_wo_stpoint(double start, double end, int size) {

        double[] out = new double[size];
        double range = end - start;
        double step_val = range / size;

        for (int i = 0; i < size; i++) {
            out[i] = start + (step_val * (i + 1));
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

    public static double[] multiply1D(double[] a, double[] b) {

        double[] c = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            c[i] += a[i] * b[i];
        }
        return c;
    }


    public static double[][] multiply2D(double[][] a, double[][] b) {

        double[][] c = new double[a.length][a[0].length];

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                c[i][j] += a[i][j] * b[i][j];
            }
        }

        return c;
    }
}
