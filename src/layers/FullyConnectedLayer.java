package layers;

import utils.Array_utils;
import utils.Utils;
import utils.global_variables;

import java.text.ParseException;
import java.util.Arrays;

/**
 * the normal Dense Layer.
 * calculates the input times the weights.
 */
public class FullyConnectedLayer extends Layer {


    public static boolean hasWeights = true;
    //double[] biases; not needed because biases are just extra weigthts.
    double[][] dweights; // gradients of weights needed if the optimizer is set.
    double[][] momentum_weights; // gradients of weights needed if the optimizer is set.
    double[][] momentum_biases; // gradients of weights needed if the optimizer is set.
    //double[] dbiases;biases; not needed because biases are just extra weigthts.
    double[][] dinputs;
    double[] dinput;
    double[][] inputs; //needed for backpropagation with batch input.
    double[] input; //needed for backpropagation with Single Input.
    double BIAS = 1;
    double BIAS_PRIME = 0;
    private double[][] weights; //weights of layer.
    // or to use other methode to upgrade weights.
    // biases befinden sich auf der letzten Ebene.
    private int n_inputs;
    private int n_neurons;


    public FullyConnectedLayer(int n_input, int n_neurons) {
        //this.weights = Utils.genRandomWeights(n_input + 1, n_neurons);
        //super.weights = new double[n_input + 1][n_neurons];
        //this.weights = super.weights;
        this.parameter_size = n_input * n_neurons + n_neurons;
        input = new double[n_input];
        inputs = new double[n_input][n_input];
        this.name = "FullyConnectedLayer";
        this.n_inputs = n_input;
        this.n_neurons = n_neurons;

    }

    /**
     * init with linspace weights to get some simulation with weights which are not zero or 1.
     *
     * @param n_input
     * @param n_neurons
     * @param Test      indicates test.
     */
    public FullyConnectedLayer(int n_input, int n_neurons, boolean Test) {
        double[][] w = Utils.genRandomWeights(n_input + 1, n_neurons);
        //has decimal precission because java messes up.
        super.weights = Array_utils.getLinspaceWeights_wo_endpoint(n_input + 1, n_neurons, -1, 1, 4);
        this.parameter_size = n_input * n_neurons + n_neurons;
        input = new double[n_input];
        inputs = new double[n_input][n_input];
        this.name = "FullyConnectedLayer";

        this.n_inputs = n_input;
        this.n_neurons = n_neurons;

    }

    /**
     * can not be initialised without the needed values.
     */
    public FullyConnectedLayer() {
        throw new RuntimeException();
    }

    public void setBIAS_ACT(String act) {
        if (act != "id") {
            this.BIAS = activation_utils.useForwardFunktion(act, this.BIAS);
            this.BIAS_PRIME = activation_utils.useBackwardFunktion(act, 1.0);
        }

    }

    public void setBIAS(double x) {
        this.BIAS = x;
    }


    /**
     * returns the biases on top of the weights.
     *
     * @return
     */
    public double[][] get_weights() {
        return this.weights;
    }

    /**
     * can get the biases on top. of the weights.
     *
     * @param w
     */

    public void setWeights(double[][] w) {
        //this.weights = new double[w.length][w[0].length];
        //this.biases = new double[w[0].length];
        this.weights = w;
        //this.biases = Utils.split_for_biases(w);
    }

    public double[][] addPRIME_BIAS(double[][] in) {
        double[][] out = new double[in.length][in[0].length + 1];
        for (int i = 0; i < in.length; i++) {
            for (int j = 0; j < in[0].length; j++) {
                out[i][j] = in[i][j];
            }
        }

        for (int i = 0; i < in.length; i++) {
            out[i][in.length] = this.BIAS_PRIME;
        }
        return out;
    }

    public double[] addPRIME_BIAS(double[] in) {
        double[] out = new double[in.length + 1];
        for (int i = 0; i < in.length; i++) {
            out[i] = in[i];
        }

        for (int i = 0; i < in.length; i++) {
            out[in.length] = this.BIAS_PRIME;
        }
        return out;
    }


    public double[][] addBIAS(double[][] in) {
        double[][] out = new double[in.length][in[0].length + 1];
        for (int i = 0; i < in.length; i++) {
            for (int j = 0; j < in[0].length; j++) {
                out[i][j] = in[i][j];
            }
        }

        for (int i = 0; i < in.length; i++) {
            out[i][in[0].length] = this.BIAS;
        }
        return out;
    }

    public double[] addBIAS(double[] in) {
        double[] out = new double[in.length + 1];
        for (int i = 0; i < in.length; i++) {
            out[i] = in[i];
        }


        out[in.length] = this.BIAS;

        return out;
    }

    /**
     * Forward Pas for the layer.
     *
     * @param inputs inputs of the layer.
     * @return computed output
     */
    public double[][] forward(double[][] inputs) throws Exception {
        double[][] outputs;
        //this.inputs = new double[inputs.length][weights.length];
        //this.inputs = Utils.clean_inputs(inputs, weights.length);

        this.inputs = inputs;
        inputs = this.addBIAS(inputs);
        System.out.println(Arrays.deepToString(inputs));
        System.out.println(Arrays.deepToString(weights));

        outputs = Utils.matmul2D(inputs, weights);

        //outputs = Utils.add_biases(outputs, biases);
        //da die erste dimension der weights die Input-shape is
        return (outputs);


    }

    /**
     * Forward Pas for the 1 layer.
     *
     * @param input Single input of the layer.
     * @return computed output
     */


    public double[] forward(double[] input) {
        double[] output = new double[input.length];
        //input = new double[weights.length - 1];
        //input = Utils.clean_input(input, weights.length - 1);

        double[][] weights_t = super.weights;

        weights_t = Utils.tranpose(weights_t);
        input = this.addBIAS(input);

        this.input = input;
        output = Utils.matmul2d_1d(weights_t, input);
        //output = Utils.add_bias(output, biases);
        return (output);

    }


    /**
     * computes Vailla SGD
     * expects delta values and computes the gradient. updates weights.
     */
    public double[][] backward(double[][] output_gradient, double learning_rate) throws Exception {


        double[][] t_inputs = Utils.tranpose(this.inputs);
        dweights = Utils.matmul2D(t_inputs, output_gradient);
        dbiases = Utils.sumBiases(output_gradient);
        //  Gradient on input values.

        this.dinputs = Utils.matmul2D(output_gradient, Utils.tranpose(super.weights));

        updateBiases_self(dbiases, learning_rate);
        updateWeights_self(dweights, learning_rate);

        return (this.dinputs);
    }

    /**
     * only used if learning rate is set.
     *
     * @param dbiases       delta biases
     * @param learning_rate learning rate of the NN.
     * @throws ParseException because values are corrected with decimal precision.
     */
    private void updateBiases_self(double[] dbiases, double learning_rate) throws ParseException {

        //System.out.println(dbiases.length);
        //System.out.println(this.biases.length);
        for (int i = 0; i < this.biases.length; i++) {
            //this.biases[i] += Array_utils.roundDec(-(learning_rate * dbiases[i]), global_variables.decimal_precision);
            this.biases[i] += Array_utils.roundDec(-(learning_rate * dbiases[i]), global_variables.decimal_precision);
        }
    }

    /**
     * only used if optimizer is not set.
     *
     * @param output_gradient
     * @param learning_rate
     * @throws ParseException because values are corrected with decimal precision.
     */
    private void updateWeights_self(double[][] output_gradient, double learning_rate) throws ParseException {

        for (int i = 0; i < this.weights.length; i++) {
            for (int j = 0; j < this.weights[1].length; j++) {

                //this.weights[i][j] += Array_utils.roundDec(-(learning_rate * output_gradient[i][j]), global_variables.decimal_precision);
                this.weights[i][j] += Array_utils.roundDec(-(learning_rate * output_gradient[i][j]), global_variables.decimal_precision);

            }

        }


    }


    /**
     * backward pass of the Layer.
     *
     * @param output_gradient output gradient
     * @param learning_rate   leaning rate.
     * @return calculated delta input.
     * @throws Exception Shape Error
     */
    // should not be needed but is backward for Single Data. beacuse trained is normaly with a batch.
    public double[] backward(double[] output_gradient, double learning_rate) throws Exception {


        //System.out.println(weights.length);


        output_gradient = this.addPRIME_BIAS(output_gradient);
        double[][] weights_gradient = Utils.calcWeightGradient(output_gradient, this.input);
        //double[][]t_weights = Utils.tranpose(weights);

        //logischerweise eigentlich Transpose, da Java nicht notwendig.
        double[] dinput = Utils.dotProdukt_1D(weights, output_gradient);


        if (this.n_inputs != this.weights.length) {
            this.weights = Utils.tranpose(weights);
        }

        //updateBiases_self(output_gradient, learning_rate);
        updateWeights_self(Utils.tranpose(weights_gradient), learning_rate);


        return (dinput);

    }

    /**
     * jsut calculated the delta values.
     * updating weights is done by the optimizer.
     *
     * @param output_gradient delta inputs of the last layer.
     * @return delta inputs.
     * @throws Exception Shape Error
     */
    public double[][] backward(double[][] output_gradient) throws Exception {

        output_gradient = this.addPRIME_BIAS(output_gradient);
        this.dweights = Utils.matmul2D(Utils.tranpose(this.inputs), output_gradient);
        this.dbiases = Utils.sumBiases(output_gradient);


        //  Gradient on input values.
        double[][] weights_t = Utils.tranpose(weights);
        this.dinputs = Utils.matmul2D(output_gradient, weights_t);

        return (this.dinputs);
    }

    /**
     * backward pass of the Layer.
     *
     * @param output_gradient output gradient
     *                        updating weights is done by the optimizer.
     * @return calculated delta input.
     * @throws Exception Shape Error
     */
    // should not be needed but is backward for Single Data. beacuse trained is normaly with a batch.
    public double[] backward(double[] output_gradient) throws Exception {

        output_gradient = this.addPRIME_BIAS(output_gradient);
        this.dweights = Utils.calcWeightGradient(output_gradient, this.input);
        double[][] weights_t = Utils.tranpose(weights);
        double[] dinput = Utils.dotProdukt_1D(weights_t, output_gradient);


        this.dinput = dinput;

        return (this.dinput);

    }


}







