package utils;

import java.text.DecimalFormat;
import java.util.Arrays;

import static utils.Utils.matmul2D;

/**
 * this class helps with array operations like print or
 * linspace to help simulate thinks.
 */
public class Array_utils {


    public static double[][][][] fill(double[][][][] a, double d) {

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                for (int k = 0; k < a[0][0].length; k++) {
                    for (int l = 0; l < a[0][0][0].length; l++) {
                        a[i][j][k][l] = d;
                    }
                }
            }
        }

        return a;

    }

    public static double[] flatten(double[][][][] d) {


        int size = sumUpMult(getShape(d));
        double[] a = new double[size];

        int count = 0;
        for (int i = 0; i < d.length; i++) {
            for (int j = 0; j < d[0].length; j++) {
                for (int k = 0; k < d[0][0].length; k++) {
                    for (int l = 0; l < d[0][0][0].length; l++) {
                        a[count] = d[i][j][k][l];
                        count += 1;
                    }
                }
            }
        }

        return a;

    }

    public static double[][][][] fill(int[] shape, double d) {

        double[][][][] a = new double[shape[0]][shape[1]][shape[2]][shape[3]];

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                for (int k = 0; k < a[0][0].length; k++) {
                    for (int l = 0; l < a[0][0][0].length; l++) {
                        a[i][j][k][l] = d;
                    }
                }
            }
        }

        return a;

    }

    public static int[] getShape(double[] a) {
        return new int[]{a.length};

    }

    public static int[] getShape(double[][] a) {
        return new int[]{a.length, a[0].length};

    }

    public static int[] getShape(double[][][] a) {
        return new int[]{a.length, a[0].length, a[0][0].length};

    }

    public static int[] getShape(double[][][][] a) {
        return new int[]{a.length, a[0].length, a[0][0].length, a[0][0][0].length};

    }

    public static double[] copyArray(double[] a) {
        double[] c = zerosLike(a);

        for (int i = 0; i < c.length; i++) {
            c[i] = a[i];
        }
        return c;

    }

    public static double[][] copyArray(float[][] a) {

        double[][] c = new double[a.length][a[0].length];

        for (int i = 0; i < c.length; i++) {
            for (int j = 0; j < c[0].length; j++) {
                c[i][j] = a[i][j];
            }
        }
        return c;

    }

    public static double[][][] copyArray(float[][][] a) {

        double[][][] c = new double[a.length][a[0].length][a[0][0].length];

        for (int i = 0; i < c.length; i++) {
            for (int j = 0; j < c[0].length; j++) {
                for (int k = 0; k < c[0][0].length; k++) {
                    c[i][j][k] = a[i][j][k];
                }

            }
        }
        return c;

    }

    public static double[][] copyArray(double[][] a) {
        double[][] c = zerosLike(a);


        for (int i = 0; i < c.length; i++) {
            for (int j = 0; j < c[0].length; j++) {
                c[i][j] = a[i][j];
            }
        }
        return c;

    }

    public static double[][][] copyArray(double[][][] a) {
        double[][][] c = zerosLike(a);


        for (int i = 0; i < c.length; i++) {
            for (int j = 0; j < c[0].length; j++) {
                for (int k = 0; k < c[0][0].length; k++) {
                    c[i][j][k] = a[i][j][k];
                }

            }
        }
        return c;

    }

    public static double[][] flattenBatch(double[][][][] a) {

        int[] shape = getShape(a);

        double[][] c = new double[shape[0]][shape[1] * shape[2] * shape[3]];
        int count = 0;
        for (int fi = 0; fi < shape[0]; fi++) {
            for (int i = 0; i < shape[1]; i++) {
                for (int j = 0; j < shape[2]; j++) {
                    for (int k = 0; k < shape[3]; k++) {
                        c[fi][count] = a[fi][i][j][k];
                    }
                    count += 1;
                }
            }
        }


        return c;
    }

    public static double[][][][] reFlat(double[][] a, int[] shape) {

        double[][][][] c = new double[shape[0]][shape[1]][shape[2]][shape[3]];
        int count = 0;
        for (int fi = 0; fi < shape[0]; fi++) {
            for (int i = 0; i < shape[1]; i++) {
                for (int j = 0; j < shape[2]; j++) {
                    for (int k = 0; k < shape[3]; k++) {
                        c[fi][i][j][k] = a[fi][count];
                    }
                    count += 1;
                }
            }
        }


        return c;
    }

    public static double[][][] reFlat(double[] a, int[] shape) {

        double[][][] c = new double[shape[0]][shape[1]][shape[2]];
        int count = 0;
        for (int fi = 0; fi < shape[0]; fi++) {
            for (int i = 0; i < shape[1]; i++) {
                for (int j = 0; j < shape[2]; j++) {
                    c[fi][i][j] = a[count];
                    count += 1;
                }
            }
        }


        return c;
    }

    public static double[][][][] copyArray(double[][][][] a) {
        double[][][][] c = zerosLike(a);


        for (int i = 0; i < c.length; i++) {
            for (int j = 0; j < c[0].length; j++) {
                for (int k = 0; k < c[0][0].length; k++) {
                    for (int l = 0; l < c[0][0][0].length; l++) {
                        c[i][j][k][l] = a[i][j][k][l];
                    }

                }

            }
        }
        return c;

    }

    public static double[][][][] multiply_axis1(double[][][][] a, double[] b) {


        double[][][][] c = new double[a.length][a[0].length][a[0][0].length][a[0][0][0].length];

        for (int j = 0; j < a.length; j++) {
            for (int i = 0; i < a[0].length; i++) {
                for (int k = 0; k < a[0][0].length; k++) {
                    for (int l = 0; l < a[0][0][0].length; l++) {
                        c[i][j][k][l] = a[i][j][k][l] * b[j];
                    }
                }
            }
        }
        return c;
    }

    public static double[][][][] Matrix_div_axis_1(double[][][][] a, double[] b) {


        double[][][][] c = new double[a.length][a[0].length][a[0][0].length][a[0][0][0].length];

        for (int j = 0; j < a.length; j++) {
            for (int i = 0; i < a[0].length; i++) {
                for (int k = 0; k < a[0][0].length; k++) {
                    for (int l = 0; l < a[0][0][0].length; l++) {
                        c[i][j][k][l] = a[i][j][k][l] / b[j];
                    }
                }
            }
        }
        return c;
    }

    public static double[][][] Matrix3D_div2D(double[][][] a, double[][] b) {
        double[][][] c = zerosLike(a);


        if (b[0].length == c[0][0].length) {
            for (int i = 0; i < a.length; i++) {
                for (int j = 0; j < a[0].length; j++) {
                    for (int k = 0; k < a[0][0].length; k++) {
                        c[i][j][k] = a[i][j][k] / b[j][k];
                    }

                }
            }
        } else {

            for (int i = 0; i < a.length; i++) {
                for (int j = 0; j < a[0].length; j++) {
                    for (int k = 0; k < a[0][0].length; k++) {
                        c[i][j][k] = a[i][j][k] / b[i][k];
                    }
                }
            }
        }
        return c;
    }

    public static double[][] Matrix2D_div1D(double[][] a, double[] b) {
        double[][] c = new double[a.length][a[0].length];


        if (b.length == c[0].length) {
            for (int i = 0; i < a.length; i++) {
                for (int j = 0; j < a[0].length; j++) {
                    c[i][j] = a[i][j] / b[j];
                }
            }
        } else {

            for (int i = 0; i < a.length; i++) {
                for (int j = 0; j < a[0].length; j++) {
                    c[i][j] = a[i][j] / b[i];
                }
            }
        }
        return c;
    }

    public static double[][] addMatrixScalar(double[][] a, double scalar) {

        double[][] c = zerosLike(a);
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                c[i][j] = a[i][j] + scalar;
            }

        }

        return c;
    }

    public static double[] addMatrixScalar(double[] a, double scalar) {

        double[] c = zerosLike(a);
        for (int i = 0; i < a.length; i++) {
            c[i] = a[i] + scalar;
        }

        return c;
    }

    public static double[][][][] zerosLike(double[][][][] a) {
        return new double[a.length][a[0].length]
                [a[0][0].length][a[0][0][0].length];
    }

    public static double[][][][] zerosLike(int[] a) {
        return new double[a[0]][a[1]][a[2]][a[3]];

    }

    public static double[][][] zerosLike(double[][][] a) {
        return new double[a.length][a[0].length][a[0][0].length];
    }

    public static double[][] onesLike(double[][] a) {


        double[][] c = new double[a.length][a[0].length];

        for (int i = 0; i < c.length; i++) {
            for (int j = 0; j < c[0].length; j++) {
                c[i][j] = 1;
            }
        }


        return c;
    }

    public static double[][] onesLike(int size1, int size2) {


        double[][] c = new double[size1][size2];

        for (int i = 0; i < c.length; i++) {
            for (int j = 0; j < c[0].length; j++) {
                c[i][j] = 1;
            }
        }


        return c;
    }

    public static double[][] zerosLike(double[][] a) {
        return new double[a.length][a[0].length];
    }

    public static double[] zerosLike(double[] a) {
        return new double[a.length];
    }

    public static double[][] sqrtArrayRE(double[][] a) {
        double[][] c = new double[a.length][a[0].length];

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                c[i][j] = Math.sqrt(a[i][j]);
            }
        }
        return c;
    }

    public static double[] sqrtArrayRE(double[] a) {
        double[] c = new double[a.length];

        for (int i = 0; i < a.length; i++) {
            c[i] = Math.sqrt(a[i]);
        }
        return c;
    }

    public static double[][] var_axis_0(double[][][] x) {


        double[][] mx = mean_axis_0(x);

        double[][][] c = new double[x.length][x[0].length][x[0][0].length];

        for (int j = 0; j < x.length; j++) {
            for (int i = 0; i < x[0].length; i++) {
                for (int k = 0; k < x[0][0].length; k++) {

                    c[i][j][k] = Math.pow((x[i][j][k] - mx[i][k]), 2);
                }

            }
        }

        return mean_axis_0(c);
    }

    public static double[] sum_axis_0_2_3(double[][][][] a) {

        double[] mx = new double[a[0].length];

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                for (int k = 0; k < a[0][0].length; k++) {
                    for (int l = 0; l < a[0][0][0].length; l++) {
                        mx[j] += a[i][j][k][l];
                    }
                }
            }
        }
        return mx;
    }

    public static double[] mean_axis_0_2_3(double[][][][] a) {

        double[] mx = sum_axis_0_2_3(a);

        for (int i = 0; i < mx.length; i++) {
            mx[i] /= a[0].length;
        }
        return mx;
    }

    public static double[] var_axis_0_2_3(double[][][][] a) {

        double[] mx = mean_axis_0_2_3(a);

        double[][][][] c = new double[a.length][a[0].length][a[0][0].length][a[0][0][0].length];

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                for (int k = 0; k < a[0][0].length; k++) {
                    for (int l = 0; l < a[0][0][0].length; l++) {
                        c[i][j][k][l] = Math.pow((a[i][j][k][l] - mx[j]), 2);
                    }
                }
            }
        }
        return mean_axis_0_2_3(c);
    }

    public static double[][] var_axis_1(double[][][] x) {


        double[][] mx = mean_axis_1(x);

        double[][][] c = new double[x.length][x[0].length][x[0][0].length];

        for (int j = 0; j < x.length; j++) {
            for (int i = 0; i < x[0].length; i++) {
                for (int k = 0; k < x[0][0].length; k++) {

                    c[i][j][k] = Math.pow((x[i][j][k] - mx[j][k]), 2);
                }
            }
        }
        return mean_axis_1(c);
    }

    public static double var(double[] x) {

        double mx = Utils.mean(x);
        double[] c = new double[x.length];

        for (int i = 0; i < x.length; i++) {

            c[i] = Math.pow((x[i] - mx), 2);

        }

        return Utils.mean(c);
    }

    public static double[] var_axis_0(double[][] x) {

        double[] mx = mean_axis_0(x);
        double[][] c = new double[x.length][x[0].length];

        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                c[i][j] = Math.pow((x[i][j] - mx[j]), 2);
            }
        }

        return mean_axis_0(c);
    }

    public static double[] var_axis_1(double[][] x) {


        double[] mx = mean_axis_1(x);

        double[][] c = new double[x.length][x[0].length];

        for (int j = 0; j < x.length; j++) {
            for (int i = 0; i < x[0].length; i++) {
                c[i][j] = Math.pow((x[i][j] - mx[j]), 2);
            }
        }

        return mean_axis_1(c);
    }

    public static double[] mean_axis_1(double[][] x) {
        double[] out = sum_axis_1(x);

        for (int i = 0; i < out.length; i++) {
            out[i] /= x[0].length;
        }

        return out;
    }

    public static double[] mean_axis_0(double[][] x) {
        double[] out = sum_axis_0(x);

        System.out.println(out.length);
        for (int i = 0; i < out.length; i++) {
            out[i] /= x.length;
        }

        return out;
    }

    public static double[][] mean_axis_1(double[][][] x) {
        double[][] out = sum_axis_1(x);

        for (int i = 0; i < out.length; i++) {
            for (int j = 0; j < out[0].length; j++) {
                out[i][j] /= x[0].length;
            }

        }
        return out;
    }

    public static double[][] mean_axis_0(double[][][] x) {
        double[][] out = sum_axis_0(x);

        for (int i = 0; i < out.length; i++) {
            for (int j = 0; j < out[0].length; j++) {
                out[i][j] /= x.length;
            }

        }

        return out;
    }

    public static double[] neg_sum_axis_0(double[][] x) {
        double[] out = new double[x[0].length];
        for (int j = 0; j < x.length; j++) {
            for (int i = 0; i < x[0].length; i++) {
                out[i] += -x[j][i];
            }
        }

        return out;
    }

    public static double[] neg_sum_axis_0_2_3(double[][][][] x) {
        double[] out = new double[x[0].length];
        for (int j = 0; j < x.length; j++) {
            for (int i = 0; i < x[0].length; i++) {
                for (int k = 0; k < x[0][0].length; k++) {
                    for (int l = 0; l < x[0][0][0].length; l++) {
                        out[j] += -x[j][i][k][l];
                    }

                }
            }
        }

        return out;
    }

    public static double[][] neg_sum_axis_0(double[][][] x) {
        double[][] out = new double[x[0].length][x[0][0].length];
        for (int j = 0; j < x.length; j++) {
            for (int i = 0; i < x[0].length; i++) {
                for (int k = 0; k < x[0][0].length; k++) {
                    out[i][k] += -x[j][i][k];
                }
            }
        }

        return out;
    }

    public static double[] sum_axis_0(double[][] x) {
        double[] out = new double[x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                out[j] += x[i][j];
            }
        }

        return out;
    }

    public static double[] sum_axis_1(double[][] x) {
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                out[i] += x[i][j];
            }
        }

        return out;
    }

    public static double[][] sum_axis_0(double[][][] x) {
        double[][] out = new double[x[0].length][x[0][0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                for (int k = 0; k < x[0][0].length; k++) {
                    out[j][k] += x[i][j][k];
                }

            }
        }

        return out;
    }

    public static double[][] sum_axis_1(double[][][] x) {
        double[][] out = new double[x[0].length][x[0][0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                for (int k = 0; k < x[0][0].length; k++) {
                    out[i][k] += x[i][j][k];
                }

            }
        }
        return out;
    }

    public static double[] addMatrixRE(double[] a, double[] b) {
        double[] c = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            c[i] = a[i] + b[i];
        }
        return c;


    }

    public static void addMatrix(double[] a, double[] b) {
        for (int i = 0; i < a.length; i++) {
            a[i] += b[i];
        }
    }

    public static void addMatrix(double[][] a, double[][] b) {
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                a[i][j] += b[i][j];
            }
        }


    }

    public static double[][] addMatrixRE(double[][] a, double[][] b) {
        double[][] c = zerosLike(a);
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                c[i][j] = a[i][j] + b[i][j];
            }
        }

        return c;
    }

    public static void addMatrix(double[][][] a, double[][][] b) {
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                for (int k = 0; k < a[0][0].length; k++) {
                    a[i][j][k] += b[i][k][j];
                }
            }
        }


    }

    public static void addMatrix(double[][][] a, double[][] b) {
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                for (int k = 0; k < a[0][0].length; k++) {
                    a[i][j][k] += b[i][j];
                }
            }
        }


    }

    public static double[][][] addMatrixRE(double[][][] a, double[][][] b) {
        double[][][] c = new double[a.length][a[0].length][a[0][0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                for (int k = 0; k < a[0][0].length; k++) {
                    c[i][j][k] = a[i][j][k] + b[i][k][j];
                }
            }
        }
        return c;

    }

    public static double[][] subMatrixRE(double[][] a, double[][] b) {
        double[][] c = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a.length; j++) {
                c[i][j] = a[i][j] - b[i][j];
            }
        }

        return c;


    }

    public static void subMatrix(double[][] a, double[][] b) {
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a.length; j++) {
                a[i][j] -= b[i][j];
            }
        }


    }

    public static double[] mult_matrix_by_scalar(double[] a, double scalar) {
        double[] c = new double[a.length];

        for (int i = 0; i < a.length; i++) {
            c[i] = a[i] * scalar;
        }
        return c;
    }

    public static void mult_matrix_by_scalar(double[][][] a, double scalar) {
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                for (int k = 0; k < a[0][0].length; k++) {
                    a[i][j][k] *= scalar;
                }
            }
        }

    }

    public static double[][] mult_matrix_by_scalarRE(double[][] a, double scalar) {

        double[][] c = new double[a.length][a[0].length];

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {

                a[i][j] *= scalar;
            }
        }

        return c;
    }

    public static double[] div_matrix_by_scalarRe(double[] a, double scalar) {

        double[] c = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            c[i] = a[i] / scalar;

        }

        return c;
    }

    public static void div_matrix_by_scalar(double[] a, double scalar) {

        for (int i = 0; i < a.length; i++) {
            a[i] /= scalar;

        }


    }

    public static void div_matrix_by_scalar(double[][] a, double scalar) {

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                a[i][j] /= scalar;
            }
        }


    }

    public static double[][][] div_matrix_by_scalarRE(double[][][] a, double scalar) {

        double[][][] c = zerosLike(a);
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                for (int k = 0; k < a[0][0].length; k++) {
                    c[i][j][k] = a[i][j][k] / scalar;
                }

            }
        }

        return c;

    }

    public static double[][] div_matrix_by_scalarRE(double[][] a, double scalar) {

        double[][] c = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                c[i][j] = a[i][j] / scalar;
            }
        }

        return c;

    }

    public static double[] div_matrix_by_scalarRE(double[] a, double scalar) {

        double[] c = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            c[i] = a[i] / scalar;

        }

        return c;

    }

    public static void div_matrix_by_scalar(double[][][] a, double scalar) {
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                for (int k = 0; k < a[0][0].length; k++) {
                    a[i][j][k] /= scalar;
                }
            }
        }

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

    public static double[][] getSubmatrix(double[][] in, int h_st, int h_end, int w_st, int w_end) {

        double[][] out = new double[h_end - h_st][w_end - w_st];

        for (int i = 0; i < h_end - h_st; i++) {
            for (int j = 0; j < w_end - w_st; j++) {
                out[i][j] = in[h_st + i][w_st + j];
            }
        }
        return out;
    }

    public static double[][][] getSubmatrix(double[][][] in, int h_st, int h_end, int w_st, int w_end, int ci) {

        double[][][] out = new double[h_end - h_st][w_end - w_st][1];

        for (int i = 0; i < h_end - h_st; i++) {
            for (int j = 0; j < w_end - w_st; j++) {
                out[i][j][0] = in[h_st + i][w_st + j][ci];
            }
        }
        return out;
    }

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

    public static double[] linspace(double start, double end, int size) {

        double[] out = new double[size];
        double range = end - start;
        double step_val = range / (size - 1);


        for (int i = 0; i < size; i++) {
            out[i] = start + (step_val * i);
        }


        return out;


    }

    public static double[][][] matmul3D(double[][][] a, double[][][] b) throws ArithmeticException {

        if (a.length != b.length || a[0][0].length != b[0].length) {
            throw new ArithmeticException();
        }

        double[][][] c = new double[a.length][a[0].length][b[0][0].length];
        for (int i = 0; i < a.length; i++) {
            c[i] = matmul2D(a[i], b[i]);
        }

        return c;
    }

    public static double[][][] matmul3D(double[][][] a, double[][][] b, String transpose) throws Exception {

        if (a.length != b.length || a[0][0].length != b[0].length) {
            throw new Exception();
        } else if (transpose.equals("b")) {
            b = reshape3D_last(b);
        } else if (transpose.equals("a")) {
            a = reshape3D_last(a);
        }


        double[][][] c = new double[a.length][a[0].length][b[0][0].length];
        for (int i = 0; i < a.length; i++) {
            c[i] = matmul2D(a[i], b[i]);
        }

        return c;
    }

    public static double[][][] reshape3D_last(double[][][] a) {

        double[][][] c = new double[a.length][a[0][0].length][a[0].length];

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                for (int k = 0; k < a[0][0].length; k++) {
                    c[i][k][j] = a[i][j][k];
                }
            }
        }
        return c;
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
     * linspace without endpoint
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

    public static double[][][][] multiply4D(double[][][][] a, double[][][][] b) {

        double[][][][] c = new double[a.length][a[0].length][a[0][0].length][a[0][0][0].length];

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                for (int k = 0; k < a[0][0].length; k++) {
                    for (int l = 0; l < a[0][0][0].length; l++) {
                        c[i][j][k][l] += a[i][j][k][l] * b[i][j][k][l];
                    }
                }

            }
        }

        return c;
    }

    public static boolean checkMultiShape(double[][][] a, double[][][] b) {


        if (a.length > b.length) {
            if (!(a.length != 1 && b.length != 1 && a.length % b.length == 0)) {
                return false;
            } else if (!(a[0].length != 1 && b[0].length != 1 && a[0].length % b[0].length == 0)) {

                return false;
            } else if (!(a[0].length != 1 && b[0].length != 1 && a[0].length % b[0].length == 0)) {
                return false;
            }


        }


        return true;
    }

    public static double[][][] multiply3D(double[][][] a, double[][][] b) {

        double[][][] c = new double[a.length][a[0].length][a[0][0].length];


        if (getShape(a) == getShape(b)) {

            for (int i = 0; i < a.length; i++) {
                for (int j = 0; j < a[0].length; j++) {
                    for (int k = 0; k < a[0][0].length; k++) {
                        c[i][j][k] += a[i][j][k] * b[i][j][k];
                    }

                }
            }
        } else if (!checkMultiShape(a, b)) {
            throw new ArithmeticException("Shape Error Got Shape a: " + Arrays.toString(getShape(a)) + " and shape b: " + Arrays.toString(getShape(b)));
        } else if (b.length == 1 && b[0].length == a[0].length && a[0][0].length == b[0][0].length) {
            for (int i = 0; i < a.length; i++) {
                for (int j = 0; j < a[0].length; j++) {
                    for (int k = 0; k < a[0][0].length; k++) {
                        c[i][j][k] += a[i][j][k] * b[0][j][k];
                    }

                }
            }
        }

        return c;
    }

    public static double sum(double[] a) {

        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i];
        }
        return sum;

    }

    public static double[][] flipud_fliplr(double[][] a) {

        int s1 = a.length;
        int s2 = a[0].length;

        double[][] c = new double[s1][s2];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                c[i][j] = a[s1 - 1 - i][s1 - 1 - j];
            }
        }
        return c;
    }

    public static double[] copyArray(float[] floats) {

        double[] c = new double[floats.length];
        for (int i = 0; i < floats.length; i++) {
            c[i] = floats[i];
        }

        return c;
    }

    public static void printShape(double[][][] a) {

        System.out.println(Arrays.toString(getShape(a)));

    }

    public static void printShape(double[][] a) {

        System.out.println(Arrays.toString(getShape(a)));

    }

    public static void printShape(double[] a) {

        System.out.println(Arrays.toString(getShape(a)));

    }

    public static void printShape(int[] a) {

        System.out.println(Arrays.toString(a));

    }

    public static int sumUpMult(int[] a) {

        int sum = a[0];
        for (int i = 1; i < a.length; i++) {
            sum *= a[i];
        }

        return sum;
    }

    public static double sumUpMult(double[] a) {

        double sum = a[0];
        for (int i = 1; i < a.length; i++) {
            sum *= a[i];
        }

        return sum;
    }

    public static int getFlattenInputShape(int[] shape) {

        int sum = shape[0];
        for (int i = 1; i < shape.length; i++) {
            sum *= shape[i];
        }

        return sum;
    }

    public void printShape(double[][][][] a) {

        System.out.println(Arrays.toString(getShape(a)));

    }


}


