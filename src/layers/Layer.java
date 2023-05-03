package layers;

public class Layer {

    public boolean hasWeights = false;

    public int parameter_size = 0;


    public double[][] weights;
    public double[][] dweights;
    public double[][] momemtum_weights;
    public double[] biases;
    public double[] dbiases;
    public double[] momentum_biases;
    public String name = "Layer";
    public double BIAS = 0;
    public double BIAS_PRIME = 0;

    public double[] forward(double[] input) {
        return input;
    }

    public void setBIAS(double x) {
        this.BIAS = x;
    }

    public double[][] forward(double[][] inputs) throws Exception {
        return inputs;
    }


    public double[] backward(double[] dinput) throws Exception {
        return dinput;
    }

    public double[] backward(double[] dinput, double learning_rate) throws Exception {
        return dinput;
    }

    public double[][] backward(double[][] dinputs) throws Exception {
        return dinputs;
    }

    public double[][] backward(double[][] dinputs, double learning_rate) throws Exception {
        return dinputs;
    }


    public void setWeights(double[][] a) {
        this.weights = a;
    }


}
