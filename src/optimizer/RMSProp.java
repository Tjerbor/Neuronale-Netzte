package optimizer;


/**
 * Dies ist ein weitere Optimizer, um die Weights noch effizienter zu aktualisieren.
 * ähnlich wie bei Momentum werden hier die vorherigen weighs
 * genau wie bei Adam ist diese Klasse zur demonstration und nicht für einen gezielten layer
 * gestalt.
 */
public class RMSProp implements Optimizer {


    double alpha = 0.001; //is the learning Rate

    double beta2 = 0.9; //
    double epsilon = 1e-8; //normalize Value


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
     * returns the new weights the new delta prev
     *
     * @param w       the weights of the given Layer
     * @param dw      the deltaWeights/gradientWeight of the Layer.
     * @param dw_prev the previous deltaWeights of the Layer.
     * @return
     */


    public double[][][] updateParameter(double[][] w, double[][] dw, double[][] dw_prev) {

        double[][][] out = new double[2][][];

        double[][] dwPrevNew = new double[w.length][w[0].length];
        double[][] wNew = new double[w.length][w[0].length];
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[0].length; j++) {
                dwPrevNew[i][j] = (beta2 * dw_prev[i][j]) + ((1 - beta2) * Math.pow(dw[i][j], 2));
                wNew[i][j] = w[i][j] - alpha * (dw[i][j] / (Math.pow(dwPrevNew[i][j], 0.5) + epsilon));

            }
        }


        out[0] = wNew;
        out[1] = dwPrevNew;
        return out;

    }

    public double[][] updateParameter(double[] w, double[] dw, double[] dw_prev) {

        double[][] out = new double[2][];

        double[] dwPrevNew = new double[w.length];
        double[] wNew = new double[w.length];
        for (int i = 0; i < w.length; i++) {
            dwPrevNew[i] = (beta2 * dw_prev[i]) + ((1 - beta2) * Math.pow(dw[i], 2));
            wNew[i] = w[i] - alpha * (dw[i] / (Math.pow(dwPrevNew[i], 0.5) + epsilon));


        }


        out[0] = wNew;
        out[1] = dwPrevNew;
        return out;

    }

    public double[][][][][] updateParameter(double[][][][] w, double[][][][] dw, double[][][][] dw_prev) {

        double[][][][][] out = new double[2][][][][];

        double[][][][] dwPrevNew = new double[w.length][w[0].length][w[0][0].length][w[0][0][0].length];
        double[][][][] wNew = new double[w.length][w[0].length][w[0][0].length][w[0][0][0].length];
        ;
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[0].length; j++) {
                for (int k = 0; k < w[0][0].length; k++) {
                    for (int l = 0; l < w[0][0][0].length; l++) {
                        dwPrevNew[i][j][k][l] = (beta2 * dw_prev[i][j][k][l]) + ((1 - beta2) * Math.pow(dw[i][j][k][l], 2));
                        wNew[i][j][k][l] = w[i][j][k][l] - alpha * (dw[i][j][k][l] / (Math.pow(dwPrevNew[i][j][k][l], 0.5) + epsilon));
                    }
                }


            }
        }


        out[0] = wNew;
        out[1] = dwPrevNew;
        return out;

    }


}
