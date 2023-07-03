package utils;


import exceptionMsg.MatrixExceptions;

import java.util.Arrays;
import java.util.Iterator;

import static load.writeUtils.writeShape;

public class Matrix<T> implements Iterable {

    T data;

    int dim;

    int[] shape;


    public Matrix(Matrix m) {
        this.shape = m.shape;
        this.dim = m.getDim();
        this.data = (T) m.getData();

    }

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


    @Override
    public Iterator iterator() {
        return new MyIterator();
    }

    public Object getItem(int pos) {

        if (this.dim == 0) {
            return data;
        } else if (this.dim == 2) {
            return getArrayPos2D(getData2D(), pos);
        } else if (this.dim == 3) {
            return getArrayPos3D(getData3D(), pos, shape);
        } else if (this.dim == 4) {
            return getArrayPos4D(getData4D(), pos, shape);

        }
        return null;
    }

    //TODO needs to be tested. 
    private Object getArrayPos5D(double[][][][][] d, int pos, int[] shape) {


        int z = 0;

        while (pos >= shape[1] * shape[2] * shape[3] * shape[4]) {
            z += 1;
            pos -= shape[1] * shape[2] * shape[3] * shape[4];
        }


        int[] shapeT = new int[]{shape[1], shape[2], shape[3], shape[4]};

        return getArrayPos4D(d[z], pos, shapeT);


    }

    private Object getArrayPos4D(double[][][][] d, int pos, int[] shape) {


        int z = 0;

        while (pos >= shape[1] * shape[2] * shape[3]) {
            z += 1;
            pos -= shape[1] * shape[2] * shape[3];
        }


        int[] shapeT = new int[]{shape[1], shape[2], shape[3]};

        return getArrayPos3D(d[z], pos, shapeT);


    }

    private Object getArrayPos3D(double[][][] d, int pos, int[] shape) {

        int z = 0;


        while (pos >= shape[1] * shape[2]) {
            z += 1;
            pos -= shape[1] * shape[2];
        }
        return getArrayPos2D(d[z], pos, shape[2]);


    }

    private Object getArrayPos2D(double[][] d, int pos) {

        int maxY;
        maxY = shape[1];
        int x = 0;

        while (true) {
            if (pos >= maxY) {
                pos -= maxY;
                x += 1;
            } else {
                break;
            }
        }
        return d[x][pos];


    }

    private Object getArrayPos2D(double[][] d, int pos, int shapeY) {

        int maxX;
        maxX = shapeY;
        int x = 0;


        //means is greater than x.
        if (pos < maxX) {
            return d[0][pos];
        } else {
            while (true) {
                if (pos >= maxX) {
                    pos -= maxX;
                    x += 1;
                } else {
                    break;
                }
            }

            return d[x][pos];
        }

    }

    private Object getArrayPos(int pos, int dim) {

        int max = Array_utils.sumUpMult(shape);
        if (max < pos) {
            throw new IllegalArgumentException("given position is greater than array size.");
        }

        if (dim == 3) {
            return this.getArrayPos3D(getData3D(), pos, shape);
        } else if (dim == 2) {
            return getArrayPos2D(getData2D(), pos, shape[0]);
        } else if (dim == 1) {
            Object[] o = (Object[]) data;
            return o[pos];
        } else if (dim == 0) {
            return data;
        } else {
            throw new IllegalArgumentException("unsupported Dimension " + dim);
        }

    }

    private class MyIterator implements Iterator {

        int currPos = 0;
        int max;

        public MyIterator() {
            max = Array_utils.sumUpMult(shape);
        }

        @Override
        public boolean hasNext() {

            //because index starts with zero
            if (max <= currPos) {
                return false;
            } else if (getItem(currPos) == null) {
                return false;
            }
            return true;
        }

        @Override
        public Object next() {
            Object o = getItem(currPos);

            currPos += 1;

            return o;

        }
    }

}
