package layer;

import utils.Array_utils;

public class Flatten<E> {

    boolean batch = false;
    int[] shape;
    int dim;

    public Flatten(boolean batch) {
        this.batch = batch;
    }

    public static double[] flatten(double[][] a) {
        double[] b = new double[a.length * a[0].length];

        int c = 0;
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                b[c] = a[i][j];
                c += 1;
            }
        }
        return b;
    }

    public static double[] flatten(double[][][] a) {
        double[] b = new double[a.length * a[0].length * a[0][0].length];
        int c = 0;
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                for (int k = 0; k < a[0][0].length; k++) {
                    b[c] = a[i][j][k];
                    c += 1;
                }


            }
        }
        return b;
    }


    public static double[] flatten(double[][][][] a) {
        double[] b = new double[a.length * a[0].length * a[0][0].length];
        int c = 0;
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                for (int k = 0; k < a[0][0].length; k++) {
                    for (int l = 0; l < a[0][0][0].length; l++) {
                        b[c] = a[i][j][k][l];
                        c += 1;
                    }

                }


            }
        }
        return b;
    }


    public E forward(double[][][] a) {
        shape = new int[]{a.length, a[0].length, a[0][0].length};
        double[][] c = new double[a.length][a[0].length * a[0][0].length];

        if (batch) {
            return (E) Array_utils.flatten(a);
        }


        for (int i = 0; i < a.length; i++) {
            c[i] = Array_utils.flatten(a);
        }

        return (E) c;
    }

    public E forward(double[][] a) {

        this.shape = new int[]{a.length, a[0].length};
        return (E) Array_utils.flatten(a);
    }

    public E forward(double[][][][] a) {
        shape = new int[]{a.length, a[0].length, a[0][0].length, a[0][0][0].length};
        double[][] c = new double[a.length][a[0].length * a[0][0].length];

        if (batch) {
            for (int i = 0; i < c.length; i++) {
                c[i] = Array_utils.flatten(a[i]);

            }
            return (E) c;


        }
        return (E) flatten(a);
    }


    public E backward(double[] a) {

        if (this.shape.length == 3) {

            double[][][] c = new double[shape[0]][shape[1]][shape[2]];
            return (E) c;
        } else if (this.shape.length == 2) {
            double[][] c = new double[shape[0]][shape[1]];

            return (E) c;
        }

        return null;

    }


}
