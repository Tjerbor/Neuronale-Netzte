package main;

import utils.Array_utils;


/**
 * mithilfe dieser Klasse kann man den Optimizer in einen Layer einbinden.
 * Bessere Dokumentation ist in der @RMSprop Klasse und im ProjektBericht.
 */

public class RMSPropNew implements Optimizer {

    double alpha = 0.001;
    double beta2 = 0.9;
    double epsilon = 1e-8;

    double[][][][] dw_prev4D;
    double[][][] dw_prev3D;
    double[] dw_prev1D;


    double[][] dw_prev2D;


    @Override
    public void setLearningRate(double learningRate) {
        this.alpha = learningRate;
    }

    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }

    public void setBeta2(double beta2) {
        this.beta2 = beta2;
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    /**
     * @param w  weights of the given Layer
     * @param dw deltaWeights of the given layer
     *           updates the Weights automatically.
     */


    public void updateParameter(double[][] w, double[][] dw) {


        if (dw_prev2D == null) {
            dw_prev2D = Array_utils.zerosLike(w);
        }


        double[][] dwPrevNew = new double[w.length][w[0].length];
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[0].length; j++) {
                dw_prev2D[i][j] = (beta2 * dw_prev2D[i][j]) + ((1 - beta2) * Math.pow(dw[i][j], 2));
                w[i][j] = w[i][j] - alpha * (dw[i][j] / (Math.pow(dw_prev2D[i][j], 0.5) + epsilon));

            }
        }
        this.dw_prev2D = dwPrevNew;

    }

    public void updateParameter(double[] w, double[] dw) {

        if (this.dw_prev1D == null) {
            this.dw_prev1D = Array_utils.zerosLike(w);
        }


        for (int i = 0; i < w.length; i++) {
            dw_prev1D[i] = (beta2 * dw_prev1D[i]) + ((1 - beta2) * Math.pow(dw[i], 2));
            w[i] = w[i] - alpha * (dw[i] / (Math.pow(dw_prev1D[i], 0.5) + epsilon));


        }

    }

    public void updateParameter(double[][][][] w, double[][][][] dw) {


        if (this.dw_prev4D == null) {
            this.dw_prev4D = Array_utils.zerosLike(w);
        }

        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[0].length; j++) {
                for (int k = 0; k < w[0][0].length; k++) {
                    for (int l = 0; l < w[0][0][0].length; l++) {
                        dw_prev4D[i][j][k][l] = (beta2 * dw_prev4D[i][j][k][l]) + ((1 - beta2) * Math.pow(dw[i][j][k][l], 2));
                        w[i][j][k][l] = w[i][j][k][l] - alpha * (dw[i][j][k][l] / (Math.pow(dw_prev4D[i][j][k][l], 0.5) + epsilon));
                    }
                }
            }
        }

    }

    public void updateParameter(double[][][] w, double[][][] dw) {


        if (this.dw_prev3D == null) {
            this.dw_prev3D = Array_utils.zerosLike(w);
        }


        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[0].length; j++) {
                for (int k = 0; k < w[0][0].length; k++) {

                    dw_prev3D[i][j][k] = (beta2 * dw_prev3D[i][j][k]) + ((1 - beta2) * Math.pow(dw[i][j][k], 2));
                    w[i][j][k] = w[i][j][k] - alpha * (dw[i][j][k] / (Math.pow(dw_prev3D[i][j][k], 0.5) + epsilon));

                }
            }
        }

    }


}
