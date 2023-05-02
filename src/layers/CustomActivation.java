package layers;

import utils.Array_utils;

import java.util.Arrays;

/**
 * This class can have a diffrent activation function per Output-node.
 * that is why it needs to set the given input-Shape. or a String with the filled values.
 * theata values can be set.
 * Expects a given value starting from 1.
 * expects activation function in lower Case
 */
public class CustomActivation extends Activation {
    String[] acts;
    double[][] inputs;
    double[] input;
    double[] schwellenwert; //0 means has no schwellenwert.

    /**
     * just creates this function, so it knows the number of expected nodes.
     *
     * @param n_inputs number of inputs expected.
     */
    public CustomActivation(int n_inputs) {
        acts = new String[n_inputs];
        schwellenwert = new double[n_inputs];
        Arrays.fill(acts, "id");
        Arrays.fill(schwellenwert, 0);
    }

    /**
     * just creates this function, so it knows the number of expected nodes.
     *
     * @param n_inputs a string value filled with functions. the other values are filled with
     *                 the identity-Function.
     */
    public CustomActivation(String[] n_inputs) {
        Arrays.fill(acts, "id");
        acts = n_inputs;
        schwellenwert = new double[n_inputs.length];
        Arrays.fill(schwellenwert, 0);

    }

    public CustomActivation(String[] n_inputs, double theata) {
        Arrays.fill(acts, "id");
        acts = n_inputs;
        schwellenwert = new double[n_inputs.length];
        Arrays.fill(schwellenwert, theata);

    }


    /**
     * set the activation for the node.
     *
     * @param node     number starting from 1.
     * @param function activation which shall be set.
     * @throws Exception if the node number is too high
     */
    public void setUnit(int node, String function) throws Exception {
        if (node < acts.length) {
            acts[node - 1] = function.toLowerCase();
        } else if (node <= 0) {
            throw new IllegalArgumentException("node can only be 1 or greater.");
        } else {
            throw new IllegalArgumentException("");
        }

    }

    /**
     * set the activation for the node.
     *
     * @param node     number starting from 1.
     * @param function activation which shall be set.
     * @param theata   if set to semi linear is clip value for up and down.
     *                 otherwise binary decision.
     * @throws Exception if the node number is too high or to low.
     */
    public void setUnit(int node, String function, double theata) throws Exception {
        if (node < acts.length) {
            acts[node - 1] = function.toLowerCase();
            schwellenwert[node - 1] = theata;
        }

    }

    public double def(double x, int pos) {
        if (schwellenwert[pos] != 0) {
            return activation_utils.useForwardFunktion(acts[pos], x, schwellenwert[pos]);
        } else {
            return activation_utils.useForwardFunktion(acts[pos], x);
        }

    }

    public double prime(double x, int pos) {
        if (schwellenwert[pos] != 0) {
            return activation_utils.useBackwardFunktion(acts[pos], x, schwellenwert[pos]);
        } else {
            return activation_utils.useBackwardFunktion(acts[pos], x);
        }

    }

    public double[] forward(double[] input) {

        this.input = input;

        double out[] = new double[input.length];

        for (int i = 0; i < input.length; i++) {
            out[i] = this.def(input[i], i);
        }
        return out;
    }

    public double[][] forward(double[][] inputs) {


        double[][] out = new double[inputs.length][inputs[0].length];

        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                out[i][j] = this.def(inputs[i][j], j);
            }

        }
        this.inputs = inputs;

        return out;
    }

    public double[][] backward(double[][] dvalues, double learning_rate) throws Exception {


        double[][] outputs = new double[dvalues.length][dvalues[0].length];


        for (int i = 0; i < dvalues.length; i++) {
            for (int j = 0; j < dvalues[0].length; j++) {
                outputs[i][j] = this.prime(dvalues[i][j], j);
            }


            outputs = Array_utils.multiply2D(dvalues, outputs);
        }
        return outputs;
    }

    public double[][] backward(double[][] dvalues) throws Exception {

        double[][] outputs = new double[dvalues.length][dvalues[0].length];

        for (int i = 0; i < dvalues.length; i++) {
            for (int j = 0; j < dvalues[0].length; j++) {
                outputs[i][j] = this.prime(dvalues[i][j], j);
            }


            outputs = Array_utils.multiply2D(dvalues, outputs);
        }
        return outputs;
    }

    ;;

    public double[] backward(double[] dvalue, double learning_rate) throws Exception {

        double out[] = new double[dvalue.length];
        for (int i = 0; i < dvalue.length; i++) {
            out[i] = this.prime(dvalue[i], i);
        }


        out = Array_utils.multiply1D(dvalue, out);
        return out;
    }

    public double[] backward(double[] dvalue) throws Exception {
        double out[] = new double[dvalue.length];
        for (int i = 0; i < dvalue.length; i++) {
            dvalue[i] = this.prime(dvalue[i], i);
        }

        out = Array_utils.multiply1D(dvalue, out);
        return out;
    }


}
