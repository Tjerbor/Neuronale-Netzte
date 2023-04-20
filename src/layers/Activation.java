package layers;

import utils.Array_utils;

public class Activation extends Layer {

    boolean hasWeights = false;
    double[] input;
    double[][] inputs;

    public double def(double x) {
        return x;
    }

    public double prime(double x) {
        return x;
    }

    ;

    /**
     * every Activation has the same methode but different def and prime function
     * @param input
     * @return
     */
    public double[] forward(double[] input) {

        this.input = input;

        double out[] = new double[input.length];

        for (int i = 0; i < input.length; i++) {
            out[i] = this.def(input[i]);
        }
        return out;
    }

    public double[][] forward(double[][] inputs) {


        double[][] out = new double[inputs.length][inputs[0].length];

        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                out[i][j] = this.def(inputs[i][j]);
            }

        }
        this.inputs = inputs;

        return out;
    }

    public double[][] backward(double[][] dvalues, double learning_rate) throws Exception {


        double[][] outputs = new double[dvalues.length][dvalues[0].length];


        for (int i = 0; i < dvalues.length; i++) {
            for (int j = 0; j < dvalues[0].length; j++) {
                outputs[i][j] = this.prime(dvalues[i][j]);
            }


            outputs = Array_utils.multiply2D(dvalues, outputs);
        }
        return outputs;
    }

    public double[][] backward(double[][] dvalues) throws Exception {

        double[][] outputs = new double[dvalues.length][dvalues[0].length];

        for (int i = 0; i < dvalues.length; i++) {
            for (int j = 0; j < dvalues[0].length; j++) {
                outputs[i][j] = this.prime(dvalues[i][j]);
            }


            outputs = Array_utils.multiply2D(dvalues, outputs);
        }
        return outputs;
    }

    ;;

    public double[] backward(double[] dvalue, double learning_rate) throws Exception {

        double out[] = new double[dvalue.length];
        for (int i = 0; i < dvalue.length; i++) {
            out[i] = this.prime(dvalue[i]);
        }


        out = Array_utils.multiply1D(dvalue, out);
        return out;
    }

    public double[] backward(double[] dvalue) throws Exception {
        double out[] = new double[dvalue.length];
        for (int i = 0; i < dvalue.length; i++) {
            dvalue[i] = this.prime(dvalue[i]);
        }

        out = Array_utils.multiply1D(dvalue, out);
        return out;
    }
}





