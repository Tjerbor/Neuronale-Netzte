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

    double momentum = 0.9;
    boolean useBiases = false;
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

    private Activation act = new TanH();

    private double[][] act_inputs; //needed for backpropagation with batch input.
    private double[] act_input; //needed for backpropagation with Single Input.

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

        if (useBiases) {
            biases = random(b);
        }

    }

    public FullyConnectedLayer(int a, int b, boolean Biases) {
        if (a < 1 || b < 1) {
            throw new IllegalArgumentException("Each layer must contain at least one neuron.");
        }

        weights = new double[a][];

        for (int i = 0; i < a; i++) {
            weights[i] = random(b);
        }

        this.useBiases = Biases;
        if (useBiases) {
            biases = random(b);
        }

    }

    public FullyConnectedLayer(int a, int b, String act) {
        if (a < 1 || b < 1) {
            throw new IllegalArgumentException("Each layer must contain at least one neuron.");
        }

        this.weights = new double[a][];

        for (int i = 0; i < a; i++) {
            this.weights[i] = random(b, 0, 1);
        }
        setActivation(act);

        if (useBiases) {
            biases = random(b, 0, 1);
        }


    }

    /**
     * This method returns an array of random values between <code>-1</code> and <code>1</code>.
     */
    private static double[] random(int length) {
        return random.doubles(length, -0.1, 0.1).toArray();
    }

    private static double[] random(int length, double mean, double std) {

        double[] a = new double[length];
        for (int i = 0; i < length; i++) {
            a[i] = random.nextGaussian(mean, std);
        }

        return a;
    }


    public void setActivation(Activation act) {
        this.act = act;
    }

    public void setActivation(String act) {
        this.act = Utils.getActivation(act);
    }

    /**
     * This method gets no input because the shape of the Momentum weights is the same as weights.
     * This method is only called to initialize the Momentum-Weights.
     */
    public void activateMomentum() {
        this.momentumWeights = new double[this.weights.length][this.weights[0].length];

        if (useBiases) {
            this.momentumBiases = new double[this.biases.length];
        }

        this.useMomentum = true;


    }

    public void activateBiases() {
        this.biases = new double[this.weights[0].length];
        this.useBiases = true;
        biases = random(this.weights[0].length);

        if (useMomentum) {
            this.momentumBiases = new double[this.biases.length];
        }

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
        if (useBiases) {
            return weights.length * weights[0].length + weights[0].length;
        }
        return weights.length * weights[0].length;

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
     * only used if learning rate is set.
     *
     * @param dbiases       delta biases
     * @param learning_rate learning rate of the NN.
     * @throws ParseException because values are corrected with decimal precision.
     */
    private void updateBiases_self(double[] dbiases, double learning_rate) {

        for (int i = 0; i < this.biases.length; i++) {
            this.biases[i] -= (learning_rate * dbiases[i]);
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
                this.weights[i][j] -= (learning_rate * output_gradient[i][j]);

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


    public double[][] forwardNew(double[][] inputs) {
        double[][] result = new double[inputs.length][weights[0].length];
        act_inputs = new double[inputs.length][weights[0].length];

        this.inputs = (inputs);
        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                for (int k = 0; k < weights.length; k++) {
                    result[i][j] += inputs[i][k] * weights[k][j];
                }

                if (useBiases) {
                    result[i][j] += biases[j];
                }

                act_inputs[i][j] = result[i][j];
                result[i][j] = act.definition(result[i][j]);
            }

        }


        return result;
    }

    public double[] forward(double[] input) {
        double[] z = new double[weights[0].length];
        double[] out = new double[weights[0].length];
        this.input = input;

        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                z[j] += input[i] * weights[i][j];
            }
        }

        if (useBiases) {
            for (int j = 0; j < weights[0].length; j++) {
                z[j] += input[j] * biases[j];
            }
        }


        act_input = z;


        for (int j = 0; j < weights[0].length; j++) {
            out[j] = act.definition(z[j]);
        }


        return out;
    }

    public double[] backward(double[] grad, double learningRate) {
        double[] grad_out = new double[input.length];

        double d_act_out;
        double dWeight;

        for (int i = 0; i < input.length; i++) {

            double grad_sum = 0;

            for (int j = 0; j < weights[0].length; j++) {

                d_act_out = act.derivative(act_input[j]);
                dWeight = grad[j] * d_act_out * this.input[i];
                weights[i][j] -= dWeight * learningRate;
                grad_sum += grad[j] * d_act_out * weights[i][j];

            }


            grad_out[i] = grad_sum;

        }

        if (useBiases && useMomentum) {
            for (int i = 0; i < grad.length; i++) {
                momentumBiases[i] = momentum * momentumBiases[i] - grad[i] * learningRate;
                biases[i] -= momentumBiases[i];
            }
        } else if (useBiases) {
            for (int i = 0; i < grad.length; i++) {
                biases[i] -= grad[i] * learningRate;
            }

        }


        return grad_out;
    }

    public double[] backward(double[] grad) {
        double[] grad_out = new double[input.length];

        double d_act_out;
        this.dweights = new double[weights.length][weights[0].length];

        for (int i = 0; i < input.length; i++) {

            double grad_sum = 0;

            for (int j = 0; j < weights[0].length; j++) {

                d_act_out = act.derivative(act_input[j]);
                dweights[i][j] = grad[j] * d_act_out * this.input[i];
                grad_sum += grad[j] * d_act_out * weights[i][j];
            }

            grad_out[i] = grad_sum;

        }
        if (useBiases) {
            for (int i = 0; i < grad.length; i++) {
                dbiases[i] = grad[i];
            }
        }

        return grad_out;
    }

    public double[][] backward(double[][] grad) {

        double[][] grad_out = new double[grad.length][inputs.length];

        double d_act_out;
        this.dweights = new double[weights.length][weights[0].length];

        for (int bs = 0; bs < inputs.length; bs++) {

            for (int i = 0; i < inputs[0].length; i++) {

                double grad_sum = 0;

                for (int j = 0; j < weights[0].length; j++) {

                    d_act_out = act.derivative(act_input[j]);
                    dweights[i][j] += grad[bs][j] * d_act_out * this.input[i];
                    grad_sum += grad[bs][j] * d_act_out * weights[i][j];
                }

                grad_out[bs][i] = grad_sum;

            }

        }

        return grad_out;

    }

    public double[][] backward(double[][] grad, double learningRate) {

        double[][] grad_out = new double[grad.length][inputs.length];

        for (int bs = 0; bs < inputs.length; bs++) {

            this.input = inputs[bs];
            this.act_input = act_inputs[bs];
            this.backward(grad[bs], learningRate);

        }

        return grad_out;

    }


    public String toString(boolean export) {
        String s = "";

        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                if (j != weights[0].length - 1) {
                    s += weights[i][j] + ";";
                } else {
                    s += weights[i][j];
                }

            }
            if (i != weights.length - 1) {
                s += "\n";
            }
        }

        if (useBiases) {
            s += "\n";
            for (int i = 0; i < biases.length; i++) {
                if (i != biases.length - 1) {
                    s += biases[i] + ";";
                } else {
                    s += biases[i];
                }
            }
        }
        return s;
    }

}
