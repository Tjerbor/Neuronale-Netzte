package layer;


/**
 * only Loss supported right no.
 * more comming in the future.
 * Calculates the Mean Squared Error.
 * is the normal option and not the 1/2 one.
 */
public class MSE extends Losses {


    public double forward(double[] y_pred, double[] y_true) {

        int sum = 0;
        for (int i = 0; i < y_true.length; i++) {
            sum += Math.pow((y_true[i] - y_pred[i]), 2);
        }
        return (double) sum / y_true.length;


    }

    public double forward(double[][] y_pred, double[][] y_true) {


        int s = y_true.length;
        int s1 = y_true[0].length;

        double sum = 0;
        double[][] out = new double[s][s1];

        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s1; j++) {
                sum += Math.pow((y_true[i][j] - y_pred[i][j]), 2);
            }
        }

        return sum / (s1 * s);

    }

    public double[] backward(double[] y_pred, double[] y_true) {


        int s = y_true.length;
        double[] out = new double[s];
        for (int i = 0; i < s; i++) {
            out[i] = 2 * (y_pred[i] - y_true[i]) / s;

        }
        return out;
    }

    public double[][] backward(double[][] y_pred, double[][] y_true) {

        int size = y_true.length * y_true[0].length;

        int s = y_true.length;
        int s1 = y_true[0].length;
        double[][] out = new double[s][s1];
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s1; j++) {
                out[i][j] += (y_pred[i][j] - y_true[i][j]);
            }
        }


        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s1; j++) {
                out[i][j] = 2 * out[i][j] / y_true[0].length;
            }


        }

        return out;
    }


}
