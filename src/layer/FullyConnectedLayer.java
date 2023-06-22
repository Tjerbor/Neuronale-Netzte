package layer;

import optimizer.Optimizer;
import utils.Array_utils;
import utils.Utils;

import java.text.ParseException;
import java.util.Arrays;
import java.util.Random;

/**
 * This class models a fully connected layer of the neural network.
 * Each fully connected layer represents two layers of neurons or one edge layer.
 *
 * @see main.NeuralNetwork#create(int[], Activation)
 */
public class FullyConnectedLayer implements Layer {
    private static final Random random = new Random();

    /**
     * This variable contains the weights of the layer.
     */
    Activation act = new TanH(); //set to Tanh because is in most cases the desired Activation-function.
    NewSoftmax softmax = null; //needs to be handled differently
    Optimizer optimizer = null; //can be set in the layer so that every Optimizer can save one previous deltaWeights.
    boolean useMomentum = false;
    double momentum = 0.9; //was replaced through optimizer deprecated in the future.
    boolean useBiases = true;

    private double[][] weights;
    /**
     * This variable contains the biases of the layer.
     */
    private double[] biases;

    private double[][] dweights; // gradients of weights needed if the optimizer is set.
    private double[][] momentumWeights; // gradients of weights needed if the optimizer is set.
    private double[] momentumBiases; // gradients of weights needed if the optimizer is set.
    private double[][] dinputs; //needed if a build-System is created to get the output of the backpropagation.
    private double[] dinput; //needed if a build-System is created to get the output of the backpropagation.
    private double[][] inputs; //needed for backpropagation with batch input.
    private double[] input; //needed for backpropagation with Single Input.
    private double[] dbiases; //biases of layer.


    private double[][] act_inputs; //needed for backpropagation with batch input.
    private double[] act_input; //needed for backpropagation with Single Input.

    private double[] lastActInput;
    private double[] lastInput;


    /**
     * This constructor creates a fully connected layer with the given number of neurons of the two layers it models.
     * It adds a bias neuron and initializes the weights with random values between <code>-1</code> and <code>1</code>.
     * It throws an exception if either layer has a length that is less than <code>1</code>.
     */

    public FullyConnectedLayer(int a, int b) {
        if (a < 1 || b < 1) {
            throw new IllegalArgumentException("Each layer must contain at least one neuron.");
        }

        weights = new double[a][b];


        for (int i = 0; i < a; i++) {
            weights[i] = random(b);
        }

        biases = new double[b];
        if (useBiases) {
            biases = random(b);
        }


        genWeights(2);

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

        genWeights(2);
    }

    public FullyConnectedLayer(int a, int b, Activation act) {
        if (a < 1 || b < 1) {
            throw new IllegalArgumentException("Each layer must contain at least one neuron.");
        }

        this.weights = new double[a][];


        for (int i = 0; i < a; i++) {
            this.weights[i] = random(b);
        }
        Array_utils.printShape(weights);
        setActivation(act);

        if (useBiases) {
            biases = random(b);
        }

        genWeights(2);

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
            a[i] = random.nextGaussian();
        }

        return a;
    }

    public void setUseBiases(boolean useBiases) {
        this.useBiases = useBiases;
        this.biases = null;
        this.dbiases = null;
    }

    public void setSoftmax() {
        this.act = null;
        this.softmax = new NewSoftmax(); //needs to be an extra class because it is a global function.
        //it means considers all inputData and not x itself.
    }

    public void genWeights(int type) {

        Random rand = new Random();
        if (type == 0) {


            for (int i = 0; i < weights[0].length; i++) {
                for (int j = 0; j < weights.length; j++) {
                    weights[j][i] = rand.nextGaussian();
                }
                if (useBiases) {
                    biases[i] = rand.nextGaussian();
                }

            }


        } else if (type == 1) {
            for (int i = 0; i < weights[0].length; i++) {
                for (int j = 0; j < weights.length; j++) {
                    weights[j][i] = rand.nextGaussian(0, 1);
                }

                if (useBiases) {
                    biases[i] = rand.nextGaussian(0, 1);
                }

            }

        } else if (type == 2) {
            for (int i = 0; i < weights[0].length; i++) {
                for (int j = 0; j < weights.length; j++) {
                    weights[j][i] = rand.nextDouble(-0.1, 0.1);
                }
                if (useBiases) {
                    biases[i] = rand.nextDouble(-0.1, 0.1);
                }

            }

        } else if (type == 3) {
            for (int i = 0; i < weights[0].length; i++) {
                for (int j = 0; j < weights.length; j++) {
                    weights[j][i] = rand.nextDouble(-1, 1);
                }
                if (useBiases) {
                    biases[i] = rand.nextDouble(-1, 1);
                }

            }
        }

    }

    public void setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
        this.dweights = new double[weights.length][weights[0].length];
        if (useBiases) {
            this.dbiases = new double[biases.length];
        }
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
        //biases = random(this.weights[0].length);

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
        if (useBiases) {
            s.append("\n").append("[");
            for (int i = 0; i < biases.length; i++) {
                if (i != biases.length - 1) {
                    s.append(biases[i]).append(", ");
                } else {
                    s.append(biases[i]);
                }
            }
            s.append("]");
        }

        s.append("\n");


        return s.toString();
    }


    public double[][] forward(double[][] inputs) {
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

    public void setRandomWeights() {
        Random random = new Random();

        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                weights[i][j] = random.nextDouble(-1, 1);
            }
        }
    }

    public double[] forward(double[] input) {

        lastInput = input;

        double[] z = new double[weights[0].length];
        double[] out = new double[weights[0].length];
        for (int j = 0; j < weights[0].length; j++) {
            for (int i = 0; i < weights.length; i++) {

                z[j] += input[i] * weights[i][j];
            }

            if (useBiases) {
                z[j] += biases[j];
            }
        }


        lastActInput = z;

        if (this.softmax != null) {
            out = softmax.forward(lastActInput);
        } else {
            for (int j = 0; j < weights[0].length; j++) {
                out[j] = act.definition(z[j]);
            }

        }

        return out;
    }


    public double[] backward(double[] gradientInput, double learningRate) {
        if (optimizer == null) {
            return backwardWithoutOptimizer(gradientInput, learningRate);
        }
        return backwardWithOptimizer(gradientInput, learningRate);

    }

    /**
     * automatically expects to use AdamOptimizer. Only Optimizer which requires to set iteration Count.
     *
     * @param gradientInput gradientInput
     * @param learningRate  Learning Rate.
     * @param iteration     epoche of the Training Step.
     * @return
     */
    public double[] backward(double[] gradientInput, double learningRate, int iteration) {

        double[] gradientOutput = new double[weights.length];

        double gardientAct;
        double deltaWeight;
        double tmpW;
        if (useBiases) {
            dbiases = new double[biases.length];
        }

        dweights = new double[weights.length][weights[0].length];
        double[] gradAct = new double[lastActInput.length];

        if (softmax != null) {
            lastActInput = softmax.backward(lastActInput);
            //handles the softmax activation.
            gradAct = lastActInput;
        }

        for (int i = 0; i < weights.length; i++) {

            double gradientOutSum = 0;

            for (int j = 0; j < weights[0].length; j++) {

                if (softmax != null) {
                    gardientAct = lastActInput[j];

                } else {
                    gardientAct = act.derivative(lastActInput[j]);
                }

                tmpW = weights[i][j];

                deltaWeight = gradientInput[j] * gardientAct * lastInput[i];

                dweights[i][j] += deltaWeight;

                gradientOutSum += gradientInput[j] * gardientAct * tmpW;
            }

            gradientOutput[i] = gradientOutSum;
        }


        if (useBiases) {
            for (int i = 0; i < biases.length; i++) {
                dbiases[i] = gradAct[i] * gradientInput[i];
            }
            optimizer.setLearningRate(learningRate);
            optimizer.updateParameter(biases, dbiases);
        }

        optimizer.updateParameter(weights, dweights, iteration);


        return gradientOutput;

    }

    public double[] backwardWithOptimizer(double[] gradientInput, double learningRate) {

        double[] gradientOutput = new double[weights.length];

        double gardientAct;
        double deltaWeight;
        double tmpW;
        if (useBiases) {
            dbiases = new double[biases.length];
        }


        double[] gradAct = new double[lastActInput.length];

        if (softmax != null) {
            lastActInput = softmax.backward(lastActInput);
            //handles the softmax activation.
            gradAct = lastActInput;
        }

        for (int i = 0; i < weights.length; i++) {

            double gradientOutSum = 0;

            for (int j = 0; j < weights[0].length; j++) {

                if (softmax != null) {
                    gardientAct = lastActInput[j];

                } else {
                    gardientAct = act.derivative(lastActInput[j]);
                }

                tmpW = weights[i][j];

                deltaWeight = gradientInput[j] * gardientAct * lastInput[i];

                dweights[i][j] += deltaWeight;

                gradientOutSum += gradientInput[j] * gardientAct * tmpW;
            }

            gradientOutput[i] = gradientOutSum;
        }


        if (useBiases) {
            for (int i = 0; i < biases.length; i++) {
                dbiases[i] = gradAct[i] * gradientInput[i];
            }
            optimizer.setLearningRate(learningRate);
            optimizer.updateParameter(biases, dbiases);
        }

        optimizer.updateParameter(weights, dweights);


        return gradientOutput;
    }


    public double[] backwardWithoutOptimizer(double[] gradientInput, double learningRate) {

        double[] gradientOutput = new double[weights.length];

        double gradAct;
        double deltaWeight;
        double tmpW;


        if (softmax != null) {
            lastActInput = softmax.backward(lastActInput);
            //handles the softmax activation.
        }


        Array_utils.printShape(weights);
        for (int i = 0; i < weights.length; i++) {

            double gradientOutSum = 0;

            for (int j = 0; j < weights[0].length; j++) {

                if (softmax != null) {
                    gradAct = lastActInput[j];

                } else {
                    gradAct = act.derivative(lastActInput[j]);
                }

                tmpW = weights[i][j];

                deltaWeight = gradientInput[j] * gradAct * lastInput[i];

                weights[i][j] -= learningRate * deltaWeight;

                gradientOutSum += gradientInput[j] * gradAct * tmpW;
            }


            gradientOutput[i] = gradientOutSum;
        }

        if (useBiases) {
            for (int i = 0; i < biases.length; i++) {
                biases[i] -= learningRate * (lastActInput[i] * gradientInput[i]);
            }

        }


        return gradientOutput;
    }

    public double[] backward(double[] gradientInput) {

        double[] gradientOutput = new double[weights.length];

        double gardientAct;
        double deltaWeight;
        double tmpW;

        dweights = new double[weights.length][weights[0].length];

        if (softmax != null) {
            lastActInput = softmax.backward(lastActInput);
            //handles the softmax activation.
        }

        for (int i = 0; i < weights.length; i++) {

            double gradientOutSum = 0;

            for (int j = 0; j < weights[0].length; j++) {

                if (softmax != null) {
                    gardientAct = lastActInput[j];

                } else {
                    gardientAct = act.derivative(lastActInput[j]);
                }

                tmpW = weights[i][j];

                deltaWeight = gradientInput[j] * gardientAct * lastInput[i];

                this.dweights[i][j] = deltaWeight;

                gradientOutSum += gradientInput[j] * gardientAct * tmpW;
            }

            gradientOutput[i] = gradientOutSum;
        }

        if (this.optimizer != null) {
            optimizer.updateParameter(weights, dweights);
        }


        return gradientOutput;
    }

    public double[][] backward(double[][] grad) {

        double[][] grad_out = new double[grad.length][inputs[0].length];

        double d_act_out;
        this.dweights = new double[weights.length][weights[0].length];

        for (int bs = 0; bs < inputs.length; bs++) {

            for (int i = 0; i < inputs[0].length; i++) {

                double grad_sum = 0;

                for (int j = 0; j < weights[0].length; j++) {

                    d_act_out = act.derivative(act_inputs[i][j]);
                    dweights[i][j] += grad[bs][j] * d_act_out * this.inputs[bs][i];
                    grad_sum += grad[bs][j] * d_act_out * weights[i][j];
                }

                grad_out[bs][i] = grad_sum;

            }

        }
        return grad_out;

    }

    public double[][] backward(double[][] grad, double learningRate) {

        double[][] grad_out = new double[grad.length][inputs[0].length];

        double d_act_out;
        double dweight = 0;

        for (int bs = 0; bs < inputs.length; bs++) {

            for (int i = 0; i < inputs[0].length; i++) {

                double grad_sum = 0;

                for (int j = 0; j < weights[0].length; j++) {

                    d_act_out = act.derivative(act_inputs[bs][j]);
                    dweight = grad[bs][j] * d_act_out * this.inputs[bs][j];
                    weights[i][j] -= dweight * learningRate;
                    grad_sum += grad[bs][j] * d_act_out * weights[i][j];
                }

                grad_out[bs][i] = grad_sum;

            }

        }

        return grad_out;

    }


    public String toString(boolean export) {
        StringBuilder s = new StringBuilder();

        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                if (j != weights[0].length - 1) {
                    s.append(weights[i][j]).append(";");
                } else {
                    s.append(weights[i][j]);
                }

            }
            if (i != weights.length - 1) {
                s.append("\n");
            }
        }

        if (useBiases) {
            s.append("\n");
            for (int i = 0; i < biases.length; i++) {
                if (i != biases.length - 1) {
                    s.append(biases[i]).append(";");
                } else {
                    s.append(biases[i]);
                }
            }
        }
        return s.toString();
    }

}
