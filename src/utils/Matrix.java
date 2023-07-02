package utils;


import exceptionMsg.MatrixExceptions;

import java.util.Arrays;

import static load.writeUtils.writeShape;

public class Matrix<T> {

    T data;

    int dim;

    int[] shape;


    public Matrix(double[][][][] d) {
        data = (T) d;
        this.dim = 4;
        shape = new int[]{d.length, d[0].length, d[0][0].length, d[0][0][0].length};

    }

    public Matrix(double[][][] d) {
        data = (T) d;
        this.dim = 3;
        shape = new int[]{d.length, d[0].length, d[0][0].length};

    }

    public Matrix(double[][] d) {
        data = (T) d;
        this.dim = 2;
        shape = new int[]{d.length, d[0].length};

    }

    public Matrix(double[] d) {
        data = (T) d;
        this.dim = 1;
        shape = new int[]{d.length};
    }

    public Matrix(Double d) {
        data = (T) d;
        this.dim = 0;
        shape = new int[]{1};
    }

    public void add(double scalar) {

        if (this.dim == 4) {
            data = (T) Array_utils.add((double[][][][]) data, scalar);
        } else if (dim == 3) {
            data = (T) Array_utils.add((double[][][]) data, scalar);
        } else if (dim == 2) {
            double[][] tmp = (double[][]) data;
            for (int i = 0; i < tmp.length; i++) {
                for (int j = 0; j < tmp[0].length; j++) {
                    tmp[i][j] += scalar;
                }
            }
            data = (T) tmp;
        } else if (dim == 1) {
            double[] tmp = (double[]) data;
            for (int i = 0; i < tmp.length; i++) {
                tmp[i] += scalar;
            }
            data = (T) tmp;

        }


    }

    public int[] getShape() {
        return shape;
    }

    public T getData() {
        return data;
    }

    public double[] getData1D() {
        return (double[]) this.data;
    }

    public double[][][] getData3D() {
        return (double[][][]) this.data;

    }

    public double[][] getData2D() {
        return (double[][]) this.data;
    }

    public double[][][][] getData4D() {
        return (double[][][][]) this.data;

    }

    public boolean isEquals(Matrix m) {

        if (this.dim != m.dim) {
            return false;
        }

        T d = this.getData();

        return Arrays.equals((Object[]) d, (Object[]) this.data);


    }

    public int getDim() {
        return dim;
    }

    @Override
    public String toString() {
        String s = "Dim: " + dim + " shape: " + writeShape(shape);
        s += "\n" + data;
        return s;
    }

    public void subDim3(double[] b) {

        double[][][] c = (double[][][]) data;
        if (shape[1] == b.length) {
            for (int i = 0; i < c.length; i++) {
                for (int j = 0; j < c[0].length; j++) {
                    for (int k = 0; k < c[0][0].length; k++) {
                        c[i][j][k] -= b[i];
                    }
                }
            }
        } else if (shape[1] == b.length) {
            for (int i = 0; i < shape[0]; i++) {
                for (int j = 0; j < shape[1]; j++) {
                    for (int k = 0; k < shape[2]; k++) {
                        c[i][j][k] -= b[i];
                    }
                }
            }
        } else if (shape[2] == b.length) {
            for (int i = 0; i < shape[0]; i++) {
                for (int j = 0; j < shape[1]; j++) {
                    for (int k = 0; k < shape[2]; k++) {
                        c[i][j][k] -= b[i];
                    }
                }
            }
        } else {
            throw new IllegalArgumentException(MatrixExceptions.mismatchingShape(this, new Matrix(b)));
        }

    }

    public Matrix copy() {

        if (this.dim == 4) {

            double[][][][] c = Array_utils.copyArray((double[][][][]) data);
            return new Matrix(c);

        } else if (dim == 3) {
            double[][][] c = Array_utils.copyArray((double[][][]) data);
            return new Matrix(c);
        } else if (dim == 2) {
            double[][] c = Array_utils.copyArray((double[][]) data);
            return new Matrix(c);
        } else if (dim == 1) {
            double[] c = Array_utils.copyArray((double[]) data);
            return new Matrix(c);

        } else if (dim == 0) {
            Double d = (Double) data;
            return new Matrix(d);
        } else {
            throw new IllegalArgumentException("Array has unsupported dimension.");
        }
    }
}
