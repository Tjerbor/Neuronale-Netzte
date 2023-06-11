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

    public static double[][][] Matrix3Ddiv2D(double[][][] a, double[][] b) {
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

    public static double[][] Matrix2Ddiv1D(double[][] a, double[] b) {
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

    public static void mult_matrix_by_scalar(double[][][] a, double scalar) {
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                for (int k = 0; k < a[0][0].length; k++) {
                    a[i][j][k] *= scalar;
                }
            }
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

    public static double[][][] matmul3D(double[][][] a, double[][][] b, String tranpose) throws Exception {

        if (a.length != b.length || a[0][0].length != b[0].length) {
            throw new Exception();
        } else if (tranpose.equals("b")) {
            b = reshape3D_last(b);
        } else if (tranpose.equals("a")) {
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
            throw new ArithmeticException("Sgape Error Got shaep a: " + Arrays.toString(getShape(a)) + " and shape b: " + Arrays.toString(getShape(b)));
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

    /**
     * RE functions mean it is required to cearte a new aary and do not overwrite.
     *
     * @param
     */


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

}


