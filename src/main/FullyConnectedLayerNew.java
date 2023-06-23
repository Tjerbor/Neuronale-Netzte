package main;

import layer.Activation;
import layer.NewSoftmax;
import layer.TanH;
import optimizer.Optimizer;
import utils.RandomUtils;
import utils.Utils;

import java.util.Arrays;

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
    Optimizer optimizer = null; //can be set in the layer so that every Optimizer can save one previous deltaWeights.
    boolean useBiases = true;

    Dropout dropout;

    double learningRate = 0.1;

    int epochAt = 0;

    boolean training = false;

    private LayerNew nextLayer;
    private LayerNew previousLayer;

    private double[][] weights;
    /**
     * This variable contains the biases of the layer.
     */
    private double[] biases;

    private double[][] dweights; // gradients of weights needed if the optimizer is set.
    private double[][] dinputs; //needed if a build-System is created to get the output of the backpropagation.
    private double[] dinput; //needed if a build-System is created to get the output of the backpropagation.
    private double[][] inputs; //needed for backpropagation with batch input.
    private double[] input; //needed for backpropagation with Single Input.
    private double[] dbiases; //biases of layer.


    private double[][] act_inputs; //needed for backpropagation with batch input.
    private double[] act_input; //needed for backpropagation with Single Input.

    private double[] lastActInput;
    private double[] lastInput;


    public FullyConnectedLayerNew() {
    }

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

        if (useBiases) {
            biases = new double[b];
            RandomUtils.genTypeWeights(2, biases);
        }

    }


    public FullyConnectedLayerNew(int a, int b, boolean Biases) {
        if (a < 1 || b < 1) {
            throw new IllegalArgumentException("Each layer must contain at least one neuron.");
        }

        weights = new double[a][b];

        RandomUtils.genTypeWeights(2, weights);

        this.useBiases = Biases;
        if (useBiases) {
            biases = new double[b];
            RandomUtils.genTypeWeights(2, biases);
        }


    }

    public FullyConnectedLayerNew(int a, int b, Activation act) {
        if (a < 1 || b < 1) {
            throw new IllegalArgumentException("Each layer must contain at least one neuron.");
        }

        this.weights = new double[a][b];

        if (useBiases) {
            biases = new double[b];
        }

        setActivation(act);
        genWeights(2);
    }

    @Override
    public void setTraining(boolean training) {
        this.training = training;
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

    /**
     * This method returns the weights of the layer, including the bias nodes.
     */
    @Override
    public double[][] getWeights() {
        double[][] result = Arrays.copyOf(weights, weights.length + 1);

        result[result.length - 1] = biases;

        return result;
    }

    @Override
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

        //return out;

        if (this.nextLayer != null) {
            return nextLayer.forward(out);
        } else {
            return out;
        }


    }

    /**
     * rate musste be between 1 and 0. It is a percentage value.
     *
     * @param rate
     */
    public void setDropout(double rate) {
        dropout = new Dropout(rate);

    }

    /**
     * if the size shall be 100 % sure use this methode.
     *
     * @param size
     */
    public void setDropout(int size) {
        dropout = new Dropout(size);

    }

    @Override
    public void setEpochAt(int epochAt) {
        this.epochAt = epochAt;
    }

    public void backward(double[] gradientInput) {

        double[] grad;
        if (optimizer == null) {

            grad = backwardWithoutOptimizer(gradientInput, learningRate);
        } else {
            optimizer.setEpochAt(epochAt);
            grad = backwardWithOptimizer(gradientInput, learningRate);
        }

        if (this.previousLayer != null) {
            this.previousLayer.setEpochAt(this.epochAt);
            this.previousLayer.backward(grad);
        }

    }


    @Override
    public void backward(double[] gradientInput, double learningRate) {

        double[] grad;
        if (optimizer == null) {

            grad = backwardWithoutOptimizer(gradientInput, learningRate);
        } else {
            optimizer.setEpochAt(epochAt);
            grad = backwardWithOptimizer(gradientInput, learningRate);
        }


        if (this.previousLayer != null) {
            this.previousLayer.setEpochAt(this.epochAt);
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


    public void backward(double[][] grad) {

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

        this.useOptimizer();


        if (this.previousLayer != null) {
            this.previousLayer.backward(grad_out);
        }

    }


    @Override
    public void backward(double[][] grad, double learningRate) {

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

        if (this.previousLayer != null) {
            this.previousLayer.backward(grad_out);
        }

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

}
