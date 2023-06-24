package utils;

public class Matrix<T> {

    T data;

    int dim;

    public Matrix(double[][][][] d) {
        data = (T) d;
        this.dim = 4;

    }

    public Matrix(double[][][] d) {
        data = (T) d;
        this.dim = 3;

    }

    public Matrix(double[][] d) {
        data = (T) d;
        this.dim = 2;

    }

    public Matrix(double[] d) {
        data = (T) d;
        this.dim = 1;
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


}
