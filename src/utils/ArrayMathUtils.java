package utils;

public class ArrayMathUtils {


    public static double[][] multiply(double[][] a, double[][] b) {

        double[][] c = new double[a.length][a[0].length];


        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                c[i][j] = a[i][j] * b[i][j];
            }
        }

        return c;

    }

    public static double[][] multiply(double[][] a, double scalar) {

        double[][] c = new double[a.length][a[0].length];


        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                c[i][j] = a[i][j] * scalar;
            }
        }

        return c;

    }

    public static double[][] add(double[][] a, double[][] b) {

        double[][] c = new double[a.length][a[0].length];


        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                c[i][j] = a[i][j] + b[i][j];
            }
        }

        return c;

    }


}
