package layers;

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
