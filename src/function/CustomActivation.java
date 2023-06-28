package function;

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
    double[] schwellenwert; //0 means has no schwellenwert.

    int pos = 0;
    int back_pos = 0;

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

    public CustomActivation(int n_inputs, double theata) {
        acts = new String[n_inputs];
        schwellenwert = new double[n_inputs];
        Arrays.fill(acts, "id");
        Arrays.fill(schwellenwert, theata);
    }

    /**
     * just creates this function, so it knows the number of expected nodes.
     *
     * @param n_inputs a string value filled with functions. the other values are filled with
     *                 the identity-Function.
     */
    public CustomActivation(String[] n_inputs) {
        acts = new String[n_inputs.length];
        Arrays.fill(acts, "id");
        acts = n_inputs;
        for (int i = 0; i < n_inputs.length; i++) {
            if (!n_inputs[i].equals("")) {
                acts[i] = n_inputs[i];
            }
        }

        schwellenwert = new double[n_inputs.length];
        Arrays.fill(schwellenwert, 0);

    }

    public CustomActivation(String[] n_inputs, double theata) {
        acts = new String[n_inputs.length];
        Arrays.fill(acts, "id");
        acts = n_inputs;
        for (int i = 0; i < n_inputs.length; i++) {
            if (!n_inputs[i].equals("")) {
                acts[i] = n_inputs[i];
            }
        }
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

    @Override
    public double definition(double x) {

        double d;
        if (schwellenwert[this.pos] != 0) {

            d = activation_utils.useForwardFunktion(acts[pos], x, schwellenwert[pos]);


        } else {
            d = activation_utils.useForwardFunktion(acts[pos], x);
        }

        this.pos += 1;

        if (pos == schwellenwert.length) {
            pos = 0;
        }

        return d;


    }

    @Override
    public double derivative(double x) {
        double d;
        if (schwellenwert[this.pos] != 0) {

            d = activation_utils.useBackwardFunktion(acts[pos], x, schwellenwert[pos]);


        } else {
            d = activation_utils.useBackwardFunktion(acts[pos], x);
        }

        this.pos += 1;

        if (pos == schwellenwert.length) {
            pos = 0;
        }

        return d;


    }

    @Override
    public String toString() {
        return "custom_act";
    }
}
