package layer;

import utils.Array_utils;
import utils.Utils;

import java.text.ParseException;

/**
 * This class models a fully connected layer of the neural network.
 * Each fully connected layer represents two layers of neurons.
 *
 * @see main.NeuralNetwork#create(int[], String)
 */
public class FullyConnectedLayer implements Layer {
    private double[] biases;
    private double[][] dweights; // gradients of weights needed if the optimizer is set.
    private double[][] momentum_weights; // gradients of weights needed if the optimizer is set.
    private double[][] momentum_biases; // gradients of weights needed if the optimizer is set.
    private double[][] dinputs;
    private double[] dinput;
    private double[][] inputs; //needed for backpropagation with batch input.
    private double[] input; //needed for backpropagation with Single Input.
    private double BIAS = 1;
    private double BIAS_PRIME = 0;
    private double[][] weights; //weights of layer.
    // or to use other methode to upgrade weights.
    // biases befinden sich auf der letzten Ebene.

    public FullyConnectedLayer(int n_input, int n_neurons) {
        weights = Utils.genRandomWeights(n_input + 1, n_neurons);
        input = new double[n_input];
        inputs = new double[n_input][n_input];
    }

    /**
     * returns the biases on top of the weights.
     *
     * @return
     */
    @Override
    public double[][] getWeights() {
        return weights;
    }

    @Override
    public void setWeights(double[][] weights) {
        this.weights = Utils.split_for_weights(weights);

        biases = Utils.split_for_biases(weights);
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

    /**
     * Forward Pas for the layer.
     *
     * @param inputs inputs of the layer.
     * @return computed output
     */
    @Override
    public double[][] forward(double[][] inputs) {
        double[][] outputs;
        //this.inputs = new double[inputs.length][weights.length];
        //this.inputs = Utils.clean_inputs(inputs, weights.length);


        this.inputs = inputs;
        outputs = Utils.matmul2D(inputs, this.weights);

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
    @Override
    public double[] forward(double[] input) {
        double[] output = new double[input.length];
        //input = new double[weights.length - 1];
        //input = Utils.clean_input(input, weights.length - 1);

        double[][] weights_t = this.weights;

        weights_t = Utils.tranpose(weights_t);

        this.input = input;
        output = Utils.matmul2d_1d(weights_t, input);
        //output = Utils.add_bias(output, biases);
        return (output);

    }

    /**
     * computes Vailla SGD
     * expects delta values and computes the gradient. updates weights.
     */
    public double[][] backward(double[][] output_gradient, double learning_rate, boolean last) {


        double[][] t_inputs = Utils.tranpose(this.inputs);

        dweights = Utils.matmul2D(t_inputs, output_gradient);
        //  Gradient on input values.
        double[][] t_w = Utils.tranpose(this.weights);

        //this.dinputs = Utils.matmul2D((output_gradient), t_w);
        if (last) {
            this.dinputs = Utils.matmul2D((output_gradient), t_w);
        } else {
            this.dinputs = Utils.matmul2D(addPRIME_BIAS(output_gradient), t_w);
        }


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
    private void updateBiases_self(double[] dbiases, double learning_rate) {

        for (int i = 0; i < this.biases.length; i++) {
            //this.biases[i] += Array_utils.roundDec(-(learning_rate * dbiases[i]), global_variables.decimal_precision);
            this.biases[i] += Array_utils.roundDec(-(learning_rate * dbiases[i]), 16);
        }
    }

    /**
     * only used if optimizer is not set.
     *
     * @param output_gradient
     * @param learning_rate
     * @throws ParseException because values are corrected with decimal precision.
     */
    private void updateWeights_self(double[][] output_gradient, double learning_rate) {

        for (int i = 0; i < this.weights.length; i++) {
            for (int j = 0; j < this.weights[0].length; j++) {
                //this.weights[i][j] += Array_utils.roundDec(-(learning_rate * output_gradient[i][j]), global_variables.decimal_precision);
                this.weights[i][j] += Array_utils.roundDec(-(learning_rate * output_gradient[i][j]), 16);

            }

        }


    }

    /**
     * backward pass of the Layer.
     *
     * @param output_gradient output gradient
     * @param learning_rate   leaning rate.
     * @return calculated delta input.
     */
    // should not be needed but is backward for Single Data. beacuse trained is normaly with a batch.
    @Override
    public double[] backward(double[] output_gradient, double learning_rate) {


        output_gradient = this.addPRIME_BIAS(output_gradient);
        double[][] weights_gradient = Utils.calcWeightGradient(output_gradient, this.input);
        //double[][]t_weights = Utils.tranpose(weights);

        //logischerweise eigentlich Transpose, da Java nicht notwendig.
        double[] dinput = Utils.dotProdukt_1D(weights, output_gradient);

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
     */
    @Override
    public double[][] backward(double[][] output_gradient) {

        output_gradient = this.addPRIME_BIAS(output_gradient);
        this.dweights = Utils.matmul2D(Utils.tranpose(this.inputs), output_gradient);
        //this.dbiases = Utils.sumBiases(output_gradient);


        //  Gradient on input values.
        double[][] weights_t = Utils.tranpose(weights);
        this.dinputs = Utils.matmul2D(output_gradient, weights_t);

        return (this.dinputs);
    }

    // TODO
    @Override
    public double[][] backward(double[][] inputs, double learningRate) {
        return new double[0][];
    }

    /**
     * backward pass of the Layer.
     *
     * @param output_gradient output gradient
     *                        updating weights is done by the optimizer.
     * @return calculated delta input.
     */
    // should not be needed but is backward for Single Data. beacuse trained is normaly with a batch.
    @Override
    public double[] backward(double[] output_gradient) {

        output_gradient = this.addPRIME_BIAS(output_gradient);
        this.dweights = Utils.calcWeightGradient(output_gradient, this.input);
        double[][] weights_t = Utils.tranpose(weights);
        double[] dinput = Utils.dotProdukt_1D(weights_t, output_gradient);


        this.dinput = dinput;

        return (this.dinput);

    }
}
