public class NN_Utils {


    public static void updateParameter(double[][][] a, double[][][] b, double learningRate) {

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                for (int k = 0; k < a[0][0].length; k++) {
                    a[i][j][k] -= learningRate * b[i][j][k];
                }
            }
        }
    }

    public static void updateParameter(double[][] a, double[][] b, double learningRate) {

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                a[i][j] -= learningRate * b[i][j];
            }
        }
    }

    public static void updateParameter(double[] a, double[] b, double learningRate) {

        for (int i = 0; i < a.length; i++) {
            a[i] -= learningRate * b[i];
        }
    }

}
