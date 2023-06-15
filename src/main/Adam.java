package main;

public class Adam implements Optimizer {


    double alpha = 0.001;

    double beta1 = 0.9;
    double beta2 = 0.9;
    double epsilon = 1e-8;

    /**
     * @param w
     * @param dw
     * @param dw_prev1
     * @param dw_prev2
     * @param t        time of the loop. (iterator i)
     * @return
     */
    public double[][][] updateParameter(double[][] w, double[][] dw, double[][] dw_prev1, double[][] dw_prev2, int t) {

        double[][][] out = new double[2][][];


        double[][] dw_prev1New = new double[w.length][w[0].length];
        ;
        double dw_prev1New_corrected;
        double[][] dw_prev2New = new double[w.length][w[0].length];
        double dw_prev2New_corrected;


        double[][] wNew = new double[w.length][w[0].length];
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[0].length; j++) {
                dw_prev1New[i][j] = (beta1 * dw_prev1[i][j]) + ((1 - beta1) * dw[i][j]);
                dw_prev1New_corrected = dw_prev1New[i][j] / (1 - Math.pow(beta1, t));

                dw_prev2New[i][j] = (beta2 * dw_prev2[i][j]) + ((1 - beta2) * Math.pow(dw[i][j], 2));
                dw_prev2New_corrected = dw_prev2New[i][j] / (1 - Math.pow(beta2, t));

                wNew[i][j] = w[i][j] - alpha * (dw_prev1New_corrected / ((Math.pow(dw_prev2New_corrected, 0.5)) + epsilon));

            }
        }


        out[0] = wNew;
        out[1] = dw_prev1New;
        out[2] = dw_prev2New;
        return out;

    }


    public double[][] updateParameter(double[] w, double[] dw, double[] dw_prev1, double[] dw_prev2, int t) {

        double[][] out = new double[2][];


        double[] dw_prev1New = new double[w.length];

        double dw_prev1New_corrected;
        double[] dw_prev2New = new double[w.length];
        double dw_prev2New_corrected;


        double[] wNew = new double[w.length];
        for (int i = 0; i < w.length; i++) {
            dw_prev1New[i] = (beta1 * dw_prev1[i]) + ((1 - beta1) * dw[i]);
            dw_prev1New_corrected = dw_prev1New[i] / (1 - Math.pow(beta1, t));

            dw_prev2New[i] = (beta2 * dw_prev2[i]) + ((1 - beta2) * Math.pow(dw[i], 2));
            dw_prev2New_corrected = dw_prev2New[i] / (1 - Math.pow(beta2, t));

            wNew[i] = w[i] - alpha * (dw_prev1New_corrected / ((Math.pow(dw_prev2New_corrected, 0.5)) + epsilon));


        }


        out[0] = wNew;
        out[1] = dw_prev1New;
        out[2] = dw_prev2New;
        return out;

    }

    public double[][][][][] updateParameter(double[][][][] w, double[][][][] dw, double[][][][] dw_prev1, double[][][][] dw_prev2, int t) {

        double[][][][][] out = new double[2][][][][];


        double[][][][] dw_prev1New = new double[w.length][w[0].length][w[0][0].length][w[0][0][0].length];
        ;
        double dw_prev1New_corrected;
        double[][][][] dw_prev2New = new double[w.length][w[0].length][w[0][0].length][w[0][0][0].length];
        double dw_prev2New_corrected;


        double[][][][] wNew = new double[w.length][w[0].length][w[0][0].length][w[0][0][0].length];
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[0].length; j++) {
                for (int k = 0; k < w[0][0].length; k++) {
                    for (int l = 0; l < w[0][0][0].length; l++) {


                        dw_prev1New[i][j][k][l] = (beta1 * dw_prev1[i][j][k][l]) + ((1 - beta1) * dw[i][j][k][l]);
                        dw_prev1New_corrected = dw_prev1New[i][j][k][l] / (1 - Math.pow(beta1, t));

                        dw_prev2New[i][j][k][l] = (beta2 * dw_prev2[i][j][k][l]) + ((1 - beta2) * Math.pow(dw[i][j][k][l], 2));
                        dw_prev2New_corrected = dw_prev2New[i][j][k][l] / (1 - Math.pow(beta2, t));

                        wNew[i][j][k][l] = w[i][j][k][l] - alpha * (dw_prev1New_corrected / ((Math.pow(dw_prev2New_corrected, 0.5)) + epsilon));
                    }
                }
            }

        }


        out[0] = wNew;
        out[1] = dw_prev1New;
        out[2] = dw_prev2New;
        return out;

    }

}
