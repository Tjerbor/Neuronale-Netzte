package layer;

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

    boolean useMomentum = false;
    private double[][] weights;
    /**
     * This variable contains the biases of the layer.
     */
    private double[] biases;

    private double[][] dweights; // gradients of weights needed if the optimizer is set.
    private double[][] momentumWeights; // gradients of weights needed if the optimizer is set.
    private double[] momentumBiases; // gradients of weights needed if the optimizer is set.
    private double[][] dinputs;
    private double[] dinput;
    private double[][] inputs; //needed for backpropagation with batch input.
    private double[] input; //needed for backpropagation with Single Input.
    private double[] dbiases; //biases of layer.

    /**
     * This constructor creates a fully connected layer with the given number of neurons of the two layers it models.
     * It adds a bias neuron and initializes the weights with random values between <code>-1</code> and <code>1</code>.
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
        //biases = new double[b];
        //Arrays.fill(biases, 1);

    }

    /**
     * This method returns an array of random values between <code>-1</code> and <code>1</code>.
     */
    private static double[] random(int length) {
        return random.doubles(length, -0.1, 0.1).toArray();
    }


    /**
     * This method gets no input because the shape of the Momentum weights is the same as weights.
     * This method is only called to initialize the Momentum-Weights.
     */
    public void activateMomentum() {
        this.momentumWeights = new double[this.weights.length][this.weights[0].length];
        this.momentumBiases = new double[this.biases.length];
        this.useMomentum = true;
        //Arrays.fill(momentumBiases, 0);
        //Arrays.fill(momentumWeights[0], 0);

    }

    public double[][] getMomentumWeights() {
        return this.momentumWeights;
    }

    public double[][] getDeltaWeights() {
        return this.dweights;
    }

    public double[] getBiases() {
        return this.biases;
    }

    public double[] getDeltaBiases() {
        return this.dbiases;
    }

    public double[] getMomentumBiases() {
        return this.momentumBiases;

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
            throw new IllegalArgumentException("The array must have the correct length.");
        }

        this.weights = Arrays.copyOf(weights, weights.length - 1);

        biases = weights[weights.length - 1];
    }

    @Override
    public int parameters() {
        return weights.length * weights[0].length + weights[0].length;
    }

    @Override
    public double[] forward(double[] input) {
        this.input = (input);
        double[] result = new double[weights[0].length];

        for (int i = 0; i < weights[0].length; i++) {
            for (int j = 0; j < weights.length; j++) {
                result[i] += input[j] * weights[j][i];
            }

            result[i] += biases[i];
        }
        return result;
    }

    @Override
    public double[][] forward(double[][] inputs) {
        double[][] result = new double[inputs.length][weights[0].length];

        this.inputs = (inputs);
        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                for (int k = 0; k < weights.length; k++) {
                    result[i][j] += inputs[i][k] * weights[k][j];
                }

                result[i][j] += biases[j];
            }
        }


        return result;
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


    @Override
    public double[][] backward(double[][] output_gradient) {

        double[][] t_inputs = Utils.tranpose(this.inputs);
        dweights = Utils.matmul2D(t_inputs, output_gradient);
        double[][] t_w = Utils.tranpose(this.weights);
        this.dinputs = Utils.matmul2D((output_gradient), t_w);
        this.dbiases = Utils.sumBiases(output_gradient);

        return this.dinputs;
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
                this.weights[i][j] += (-(learning_rate * output_gradient[i][j]));

            }

        }


    }

    /**
     * This method returns a string containing the weights of the fully connected layer.
     */
    @Override
    public String toString() {
        StringBuilder s = new StringBuilder(Arrays.toString(weights[0]));

        for (int i = 1; i < weights.length; i++) {
            s.append("\n").append(Arrays.toString(weights[i]));
        }

        return s.toString();
    }
}
