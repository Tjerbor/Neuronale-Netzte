package layer;

import utils.Array_utils;
import utils.Utils;

import java.text.ParseException;
import java.util.Arrays;
import java.util.Random;

/**
 * This class models a fully connected layer of the neural network.
 * Each fully connected layer represents two layers of neurons or one edge layer.
 *
 * @see main.NeuralNetwork#create(int[], String)
 */
public class FullyConnectedLayer implements Layer {
    private static final Random random = new Random();

    /**
     * This variable contains the weights of the layer.
     */
    private double[][] weights;
    /**
     * This variable contains the biases of the layer.
     */
    private double[] biases;

    private double[][] dweights; // gradients of weights needed if the optimizer is set.
    private double[][] momentum_weights; // gradients of weights needed if the optimizer is set.
    private double[][] momentum_biases; // gradients of weights needed if the optimizer is set.
    private double[][] dinputs;
    private double[] dinput;
    private double[][] inputs; //needed for backpropagation with batch input.
    private double[] input; //needed for backpropagation with Single Input.
    private double BIAS_PRIME = 0;
    private double[] dbiases; //biases of layer.

    /**
     * This constructor creates a fully connected layer with the given number of neurons of the two layers it models.
     * It throws an exception if either layer has a length that is less than <code>1</code>.
     */
    public FullyConnectedLayer(int a, int b) {
        if (a < 1 || b < 1) {
            throw new IllegalArgumentException("Each layer must contain at least one neuron.");
        }

        weights = new double[a][];

        for (int i = 0; i < a; i++) {
            weights[i] = random(b);
        }

        biases = random(b);
    }

    /**
     * This method returns an array of random values between <code>-1</code> and <code>1</code>.
     */
    private static double[] random(int length) {
        return random.doubles(length, -1, 1).toArray();
    }

    /**
     * This method returns the weights of the layer, including the bias nodes.
     */
    @Override
    public double[][] getWeights() {
        double[][] result = Arrays.copyOf(weights, weights.length + 1);

        result[result.length - 1] = biases;

        return result;
    }

    /**
     * This method sets the weights of the layer, including the bias nodes.
     * It throws an exception if the array does not have the correct length.
     */
    @Override
    public void setWeights(double[][] weights) {
        if (weights.length != this.weights.length + 1) {
            throw new IllegalArgumentException("The array does not have the correct length.");
        }

        this.weights = Arrays.copyOf(weights, weights.length - 1);

        biases = weights[weights.length - 1];
    }

    @Override
    public int parameters() {
        return weights.length * weights[0].length + weights[0].length;
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
        this.inputs = inputs;
        double[][] outputs;
        outputs = Utils.matmul2D(inputs, this.weights);
        outputs = Utils.add_biases(outputs, biases);
        return outputs;

    }

    /**
     * Forward Pas for the 1 layer.
     *
     * @param input Single input of the layer.
     * @return computed output
     */
    @Override
    public double[] forward(double[] input) {
        this.input = input;

        double[] output;
        double[][] weightsT = this.weights;

        weightsT = Utils.tranpose(weightsT);

        output = Utils.matmul2d_1d(weightsT, input);
        output = Utils.add_bias(output, biases);
        return output;

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

        double[][] weights_gradient = Utils.calcWeightGradient(output_gradient, this.input);
        //double[][]t_weights = Utils.tranpose(weights);

        //logischerweise eigentlich Transpose, da Java nicht notwendig.
        double[] dinput = Utils.dotProdukt_1D(weights, output_gradient);


        updateBiases_self(output_gradient, learning_rate);
        updateWeights_self(Utils.tranpose(weights_gradient), learning_rate);


        return (dinput);

    }

    // TODO
    @Override
    public double[][] backward(double[][] inputs) {
        return new double[0][];
    }

    /**
     * jsut calculated the delta values.
     * updating weights is done by the optimizer.
     *
     * @param output_gradient delta inputs of the last layer.
     * @return delta inputs.
     */
    @Override
    public double[][] backward(double[][] output_gradient, double learning_rate) {

        double[][] t_inputs = Utils.tranpose(this.inputs);
        dweights = Utils.matmul2D(t_inputs, output_gradient);
        //  Gradient on input values.
        double[][] t_w = Utils.tranpose(this.weights);
        this.dinputs = Utils.matmul2D((output_gradient), t_w);
        this.dbiases = Utils.sumBiases(output_gradient);

        updateWeights_self(dweights, learning_rate);
        updateBiases_self(dbiases, learning_rate);
        return this.dinputs;
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

        this.dweights = Utils.calcWeightGradient(output_gradient, this.input);
        double[][] weightsT = Utils.tranpose(weights);
        double[] dinput = Utils.dotProdukt_1D(weightsT, output_gradient);


        this.dinput = dinput;

        return this.dinput;

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
            this.biases[i] += -(learning_rate * dbiases[i]);
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
}
