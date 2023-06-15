package main;

import utils.Array_utils;

public class AdamNew {


    double[][][][] dw_prev1_4d;
    double[][][][] dw_prev2_4d;
    double[][][] dw_prev1_3d;
    double[][][] dw_prev2_3d;

    double[][] dw_prev1_2d;
    double[][] dw_prev2_2d;

    double[] dw_prev1_1d;
    double[] dw_prev2_1d;

    double alpha = 0.001;

    double beta1 = 0.9;
    double beta2 = 0.9;
    double epsilon = 1e-8;


    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }

    public void setLearningRate(double alpha) {
        this.alpha = alpha;
    }

    /**
     * @param w
     * @param dw
     * @param t  time in the iteraton of the trainings Loop.
     */
    public void updateParameter(double[][] w, double[][] dw, int t) {


        if (dw_prev1_2d == null) {
            dw_prev1_2d = Array_utils.zerosLike(w);
            dw_prev2_2d = Array_utils.zerosLike(w);

        }


        double dw_prev1New_corrected;
        double dw_prev2New_corrected;


        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[0].length; j++) {
                dw_prev1_2d[i][j] = (beta1 * dw_prev1_2d[i][j]) + ((1 - beta1) * dw[i][j]);
                dw_prev1New_corrected = dw_prev1_2d[i][j] / (1 - Math.pow(beta1, t));

                dw_prev2_2d[i][j] = (beta2 * dw_prev2_2d[i][j]) + ((1 - beta2) * Math.pow(dw[i][j], 2));
                dw_prev2New_corrected = dw_prev2_2d[i][j] / (1 - Math.pow(beta2, t));

                w[i][j] = w[i][j] - alpha * (dw_prev1New_corrected / ((Math.pow(dw_prev2New_corrected, 0.5)) + epsilon));

            }
        }
    }


    public void updateParameter(double[] w, double[] dw, int t) {


        if (dw_prev1_1d == null) {
            dw_prev1_1d = Array_utils.zerosLike(w);
            dw_prev2_1d = Array_utils.zerosLike(w);

        }

        double dw_prev1New_corrected;
        double dw_prev2New_corrected;


        double[] wNew = new double[w.length];
        for (int i = 0; i < w.length; i++) {
            dw_prev1_1d[i] = (beta1 * dw_prev1_1d[i]) + ((1 - beta1) * dw[i]);
            dw_prev1New_corrected = dw_prev1_1d[i] / (1 - Math.pow(beta1, t));

            dw_prev2_1d[i] = (beta2 * dw_prev2_1d[i]) + ((1 - beta2) * Math.pow(dw[i], 2));
            dw_prev2New_corrected = dw_prev2_1d[i] / (1 - Math.pow(beta2, t));

            wNew[i] = w[i] - alpha * (dw_prev1New_corrected / ((Math.pow(dw_prev2New_corrected, 0.5)) + epsilon));
        }


    }

    public void updateParameter(double[][][][] w, double[][][][] dw, int t) {


        if (dw_prev1_4d == null) {
            dw_prev1_4d = Array_utils.zerosLike(w);
            dw_prev2_4d = Array_utils.zerosLike(w);

        }


        ;
        double dw_prev1New_corrected;
        double dw_prev2New_corrected;


        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[0].length; j++) {
                for (int k = 0; k < w[0][0].length; k++) {
                    for (int l = 0; l < w[0][0][0].length; l++) {


                        dw_prev1_4d[i][j][k][l] = (beta1 * dw_prev1_4d[i][j][k][l]) + ((1 - beta1) * dw[i][j][k][l]);
                        dw_prev1New_corrected = dw_prev1_4d[i][j][k][l] / (1 - Math.pow(beta1, t));

                        dw_prev2_4d[i][j][k][l] = (beta2 * dw_prev2_4d[i][j][k][l]) + ((1 - beta2) * Math.pow(dw[i][j][k][l], 2));
                        dw_prev2New_corrected = dw_prev2_4d[i][j][k][l] / (1 - Math.pow(beta2, t));

                        w[i][j][k][l] = w[i][j][k][l] - alpha * (dw_prev1New_corrected / ((Math.pow(dw_prev2New_corrected, 0.5)) + epsilon));
                    }
                }
            }

        }


    }


}
