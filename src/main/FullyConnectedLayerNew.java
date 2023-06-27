package main;

import layer.Activation;
import layer.NewSoftmax;
import layer.TanH;
import optimizer.Optimizer;
import utils.Array_utils;
import utils.Matrix;
import utils.RandomUtils;
import utils.Utils;

import java.util.Arrays;

import static utils.Array_utils.getShape;

/**
 * This class models a fully connected layer of the neural network.
 * Each fully connected layer represents two layers of neurons or one edge layer.
 *
 * @see NeuralNetwork#create(int[], Activation)
 */
public class FullyConnectedLayerNew extends LayerNew {
    /**
     * This variable contains the weights of the layer.
     */
    Activation act = new TanH(); //set to Tanh because is in most cases the desired Activation-function.
    NewSoftmax softmax = null; //needs to be handled differently
    Optimizer optimizer; //can be set in the layer so that every Optimizer can save one previous deltaWeights.
    private double[][] weights;
    /**
     * This variable contains the biases of the layer.
     */
    private double[] biases;
    private double[][] dweights; // gradients of weights needed if the optimizer is set.
    private double[] dbiases; //biases of layer.

    private double[] lastActInput;
    private double[] lastInput;

    private double[][] lastActInputs;
    private double[][] lastInputs;

    /**
     * This constructor creates a fully connected layer with the given number of neurons of the two layers it models.
     * It adds a bias neuron and initializes the weights with random values between <code>-1</code> and <code>1</code>.
     * It throws an exception if either layer has a length that is less than <code>1</code>.
     */

    public FullyConnectedLayerNew(int a, int b) {
        if (a < 1 || b < 1) {
            throw new IllegalArgumentException("Each layer must contain at least one neuron.");
        }

        weights = new double[a][b];

        RandomUtils.genTypeWeights(2, weights);

        //this.useBiases = true;
        if (useBiases) {
            biases = new double[b];
            RandomUtils.genTypeWeights(2, biases);
        }

        this.inputShape = new int[]{a};
        this.outputShape = new int[]{b};


    }

    public FullyConnectedLayerNew(int a, int b, boolean useBiases) {
        if (a < 1 || b < 1) {
            throw new IllegalArgumentException("Each layer must contain at least one neuron.");
        }

        weights = new double[a][b];

        RandomUtils.genTypeWeights(2, weights);

        this.useBiases = useBiases;
        if (this.useBiases) {
            biases = new double[b];
            RandomUtils.genTypeWeights(2, biases);
        }


    }

    public FullyConnectedLayerNew(int a, int b, Activation act) {
        if (a < 1 || b < 1) {
            throw new IllegalArgumentException("Each layer must contain at least one neuron.");
        }

        this.weights = new double[a][b];

        this.useBiases = true;
        biases = new double[b];

        setActivation(act);
        genWeights(2);
    }

    public void genWeights(int type) {
        RandomUtils.genTypeWeights(2, weights);

        if (useBiases) {
            biases = new double[weights[0].length];
            RandomUtils.genTypeWeights(2, biases);
        }
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

    public void setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
        this.dweights = new double[weights.length][weights[0].length];
        if (useBiases) {
            this.dbiases = new double[biases.length];
        }
    }

    public void useOptimizer() {

        if (useBiases) {
            this.optimizer.updateParameter(this.biases, dbiases);
        }
        this.optimizer.updateParameter(weights, dweights);
    }

    public void setActivation(Activation act) {
        this.act = act;
    }

    public void setActivation(String act) {
        this.act = Utils.getActivation(act);
    }

    public double[] getBiases() {
        return this.biases;
    }


    public Matrix getWeights() {
        Double[][] d;
        if (useBiases) {
            double[][] result = Arrays.copyOf(weights, weights.length + 1);
            result[result.length - 1] = biases;
            return new Matrix(result);

        } else {
            return new Matrix(weights);

        }


        //Stream.of(boxed).mapToDouble(Double::doubleValue).toArray();

    }

    @Override
    public void setWeights(Matrix m) {

        if (!useBiases) {
            this.weights = m.getData2D();

        } else {
            double[][] tmp = m.getData2D();
            weights = Arrays.copyOf(tmp, tmp.length - 1);
            biases = tmp[tmp.length - 1];


        }


    }


    /**
     * This method returns the weights of the layer, including the bias nodes.
     */

    public void setWeights(double[][] weights) {

        if (!useBiases) {
            this.weights = weights;

        } else {
            this.weights = Arrays.copyOf(weights, weights.length - 1);
            biases = weights[weights.length - 1];
        }
    }

    /**
     * This method sets the weights of the layer, including the bias nodes.
     * It throws an exception if the array does not have the correct length.
     */

    @Override
    public int parameters() {
        if (useBiases) {
            return weights.length * weights[0].length + weights[0].length;
        }
        return weights.length * weights[0].length;

    }


    @Override
    public int[] getInputShape() {
        return new int[]{this.weights.length};
    }


    @Override
    public int[] getOutputShape() {
        return new int[]{this.weights[0].length};
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


    public void forward(double[][] inputs) {
        double[][] result = new double[inputs.length][weights[0].length];
        lastActInputs = new double[inputs.length][weights[0].length];

        this.lastInputs = (inputs);
        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                for (int k = 0; k < weights.length; k++) {
                    result[i][j] += inputs[i][k] * weights[k][j];
                }

                if (useBiases) {
                    result[i][j] += biases[j];
                }

                lastActInputs[i][j] = result[i][j];
                result[i][j] = act.definition(result[i][j]);
            }
        }

        if (this.dropout != null) {
            result = dropout.forward(result);
        }

        if (this.getNextLayer() != null) {
            this.getNextLayer().forward(result);
        } else {
            output = new Matrix(result);
        }

    }

    @Override
    public void forward(double[][][] input) {
        double[] c = Array_utils.flatten(input);
        this.inputShape = getShape(input);
        forward(c);
    }

    @Override
    public void forward(double[][][][] inputs) {
        double[][] c = new double[inputs.length][];
        this.inputShape = getShape(inputs[0]);
        for (int i = 0; i < inputs.length; i++) {
            c[i] = Array_utils.flatten(inputs[i]);
        }

        this.forward(c);


    }

    @Override
    public LayerNew getNextLayer() {
        return this.nextLayer;
    }

    @Override
    public void setNextLayer(LayerNew l) {
        this.nextLayer = l;
    }

    @Override
    public LayerNew getPreviousLayer() {
        return this.previousLayer;
    }

    @Override
    public void setPreviousLayer(LayerNew l) {
        this.previousLayer = l;
    }

    @Override
    public void forward(double[] input) {

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

        //return out;
        if (this.dropout != null) {
            out = dropout.forward(out);
        }

        if (this.getNextLayer() != null) {
            this.getNextLayer().forward(out);
        } else {
            this.output = new Matrix(out);
        }

    }


    @Override
    public void backward(double[] gradientInput) {

        double[] grad;
        if (optimizer == null) {
            grad = backwardWithoutOptimizer(gradientInput, learningRate);
        } else {
            optimizer.setEpochAt(this.iterationAt);
            grad = backwardWithOptimizer(gradientInput, learningRate);
        }


        if (this.dropout != null) {
            grad = this.dropout.backward(grad);
        }

        if (this.previousLayer != null) {
            this.previousLayer.setIterationAt(this.iterationAt);
            this.previousLayer.backward(grad);
        }

    }


    @Override
    public void backward(double[] gradientInput, double learningRate) {

        double[] grad;
        if (optimizer == null) {
            grad = backwardWithoutOptimizer(gradientInput, learningRate);
        } else {
            optimizer.setEpochAt(iterationAt);
            grad = backwardWithOptimizer(gradientInput, learningRate);
        }

        if (this.dropout != null) {
            grad = this.dropout.backward(grad);
        }

        if (this.previousLayer != null) {
            this.previousLayer.setIterationAt(this.iterationAt);
            this.previousLayer.backward(grad, learningRate);
        }
    }


    public double[] backwardWithOptimizer(double[] gradientInput, double learningRate) {

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
        } else {
            optimizer.updateParameter(weights, dweights);
        }


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

        for (int i = 0; i < weights.length; i++) {

            double gradientOutSum = 0;

            for (int j = 0; j < weights[0].length; j++) {

                if (softmax != null) {
                    gradAct = lastActInput[j];

                } else {
                    gradAct = act.derivative(lastActInput[j]) * gradientInput[j];
                }

                tmpW = weights[i][j];

                deltaWeight = gradAct * lastInput[i];

                weights[i][j] -= learningRate * deltaWeight;

                gradientOutSum += gradientInput[j] * gradAct * tmpW;
            }


            gradientOutput[i] = gradientOutSum;
        }

        if (useBiases) {
            if (softmax != null) {
                for (int i = 0; i < biases.length; i++) {
                    biases[i] -= learningRate * (lastActInput[i]);
                }
            } else {
                for (int i = 0; i < biases.length; i++) {
                    biases[i] -= learningRate * (lastActInput[i] * gradientInput[i]);
                }
            }

        }
        return gradientOutput;
    }

    public void backward(double[][] grad) {

        double[][] grad_out = new double[lastInputs.length][lastInputs[0].length];

        double d_act_out;
        this.dweights = new double[weights.length][weights[0].length];

        for (int bs = 0; bs < lastInputs.length; bs++) {

            for (int i = 0; i < lastInputs[0].length; i++) {

                double grad_sum = 0;

                for (int j = 0; j < weights[0].length; j++) {

                    d_act_out = act.derivative(lastActInputs[i][j]);
                    dweights[i][j] += grad[bs][j] * d_act_out * this.lastInputs[bs][i];
                    grad_sum += grad[bs][j] * d_act_out * weights[i][j];
                }

                grad_out[bs][i] = grad_sum;

            }

        }

        this.useOptimizer();

        if (this.dropout != null) {
            grad_out = this.dropout.backward(grad_out);
        }

        if (this.getPreviousLayer() != null) {
            this.getPreviousLayer().backward(grad_out);
        }

    }

    @Override
    public void backward(double[][][] input) {
        double[] c;
        c = Array_utils.flatten(input);
        this.backward(c);


    }

    @Override
    public void backward(double[][][][] inputs) {
        double[][] c = new double[inputs.length][];
        for (int i = 0; i < inputs.length; i++) {
            c[i] = Array_utils.flatten(inputs[i]);
        }
        this.backward(c);
    }


    @Override
    public void backward(double[][] grad, double learningRate) {
        this.learningRate = learningRate;
        this.getNextLayer().setLearningRate(learningRate);
        this.backward(grad);
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

    /**
     * saves in maximal 3 Lines.
     *
     * @return
     */
    public String export() {
        StringBuilder s = new StringBuilder();
        String optName;
        if (this.optimizer != null) {
            optName = optimizer.export();
        } else {
            optName = "null;";
        }

        s.append("FCL;" + weights.length + ";" + weights[0].length + ";" + useBiases + ";" + optName);
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                if (j != weights[0].length - 1) {
                    s.append(weights[i][j]).append(";");
                } else {
                    s.append(weights[i][j]);
                }

            }
        }
        s.append("\n");
        if (useBiases) {
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

    @Override
    public String summary() {
        return "Fully-Connected-Layer inputSize: " + Arrays.toString(getInputShape())
                + " outputSize: " + Arrays.toString(getOutputShape())
                + " parameter: " + parameters() + "\n";
    }


    @Override
    public boolean isEqual(LayerNew other2) {

        FullyConnectedLayerNew other = (FullyConnectedLayerNew) other2;

        if (other.getInputShape() == this.inputShape && this.getWeights() == other.getWeights() && this.act == other.act) {
            return true;
        }

        return false;
    }

    @Override
    public Matrix getOutput() {
        return output;
    }
}
