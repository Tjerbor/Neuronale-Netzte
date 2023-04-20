package layers;

/**
 * loss function of the NN.
 * has the same names so it is easy to use.
 */
public class Losses {

    public double forward(double[] input, double[] y_true) {
        return 1;
    }
    public double forward(double[][] inputs, double[][] y_true) {
        return 1;
    }

    public double[] backward(double[] input, double[] y_true){return input;}
    public double[][] backward(double[][] inputs, double[][] y_true){return inputs;}

}
