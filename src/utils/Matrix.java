package utils;

import java.util.Arrays;

import static load.writeUtils.writeShape;
import static utils.Array_utils.getShape;

public class Matrix<T> {

    T data;

    int dim;

    int[] shape;

    public Matrix(double[][][][] d) {
        data = (T) d;
        this.dim = 4;
        shape = getShape(d);

    }

    public Matrix(double[][][] d) {
        data = (T) d;
        this.dim = 3;
        shape = getShape(d);

    }

    public Matrix(double[][] d) {
        data = (T) d;
        this.dim = 2;
        shape = getShape(d);

    }

    public Matrix(double[] d) {
        data = (T) d;
        this.dim = 1;
        shape = getShape(d);
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
}
