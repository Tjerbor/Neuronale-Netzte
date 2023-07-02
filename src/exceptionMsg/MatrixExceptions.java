package exceptionMsg;

import utils.Matrix;

import java.util.Arrays;

public class MatrixExceptions {

    public static String shapeError(Matrix m, Matrix m2) {
        return "matrix Multiply got Shape: m:" + Arrays.toString(m.getShape()) + " : m2: " + Arrays.toString(m2.getShape());
    }

    public static String mismatchingShape(Matrix m, Matrix m2) {
        return "matrix got Shape: m:" + Arrays.toString(m.getShape()) + " : m2: " + Arrays.toString(m2.getShape());
    }


}
