package layers;

import java.util.Arrays;

/**
 * This class can have a diffrent activation function per Output-node.
 * that is why it needs to set the given input-Shape. or a String with the filled values.
 * theata values can be set.
 * Expects a given value starting from 1.
 * expects activation function in lower Case
 */
public class CustomActivation {
    String[] acts;
    double[] schwellenwert; //0 means has no schwellenwert.

    /**
     * just creates this function, so it knows the number of expected nodes.
     * @param n_inputs number of inputs expected.
     */
    public CustomActivation(int n_inputs){
        acts = new String[n_inputs];
        schwellenwert = new double[n_inputs];
        Arrays.fill(acts, "id");
        Arrays.fill(schwellenwert, 0);
    }

    /**
     * just creates this function, so it knows the number of expected nodes.
     * @param n_inputs a string value filled with functions. the other values are filled with
     * the identity-Function.
     */
    public CustomActivation(String[] n_inputs){
        Arrays.fill(acts, "id");
        acts = n_inputs;
        schwellenwert = new double[n_inputs.length];
        Arrays.fill(schwellenwert, 0);

    }


    /**
     * set the activation for the node.
     * @param node number starting from 1.
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
     * @param node number starting from 1.
     * @param function activation which shall be set.
     * @param theata if set to semi linear is clip value for up and down.
     *  otherwise binary decision.
     * @throws Exception if the node number is too high or to low.
     */
    public void setUnit(int node, String function, double theata) throws Exception {
        if (node < acts.length) {
            acts[node - 1] = function.toLowerCase();
            schwellenwert[node - 1] = theata;
        }

    }

}
