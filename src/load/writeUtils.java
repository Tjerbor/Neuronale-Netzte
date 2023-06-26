package load;

public class writeUtils {


    public static String writeShape(int[] shape) {

        StringBuilder s = new StringBuilder();

        for (int i = 0; i < shape.length; i++) {
            if (i != shape.length - 1) {
                s.append(shape[i]).append(";");
            } else {
                s.append(shape[i]);
            }
        }
        return s.toString();
    }

    public static String writeWeights(double[] w) {

        StringBuilder s = new StringBuilder();
        for (int i = 0; i < w.length; i++) {
            if (i == w.length - 1) {
                s.append(w[i]);
            } else {

                s.append(w[i]).append(";");
            }


        }
        return s.toString();

    }

    public static String writeWeights(double[][] w) {

        StringBuilder s = new StringBuilder();
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[0].length; j++) {
                if (i == w.length - 1 && j == w[0].length - 1) {
                    s.append(w[i][j]);
                } else {

                    s.append(w[i][j]).append(";");
                }

            }
        }
        return s.toString();

    }

    public static String writeWeights(double[][][] w) {

        StringBuilder s = new StringBuilder();
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[0].length; j++) {
                for (int k = 0; k < w[0][0].length; k++) {
                    if (i == w.length - 1 && j == w[0].length - 1 && k == w[0][0].length - 1) {
                        s.append(w[i][j][k]);
                    } else {

                        s.append(w[i][j][k]).append(";");
                    }
                }


            }
        }
        return s.toString();

    }

    public static String writeWeights(double[][][][] w) {

        StringBuilder s = new StringBuilder();
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[0].length; j++) {
                for (int k = 0; k < w[0][0].length; k++) {
                    for (int l = 0; l < w[0][0][0].length; l++) {
                        if (i == w.length - 1 && j == w[0].length - 1 && k == w[0][0].length - 1 && l == w[0][0][0].length - 1) {
                            s.append(w[i][j][k][l]);
                        } else {


                            s.append(w[i][j][k][l]).append(";");
                        }
                    }

                }


            }
        }
        return s.toString();

    }
}
