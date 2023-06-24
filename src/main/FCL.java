package main;

import layer.Activation;
import layer.TanH;
import optimizer.Optimizer;
import utils.Matrix;
import utils.RandomUtils;

import java.util.Arrays;

public class FCL extends LayerNew {


    Activation act = new TanH(); //set to Tanh because is in most cases the desired Activation-function.
    Optimizer optimizer; //can be set in the layer so that every Optimizer can save one previous deltaWeights.
    private double[][] weights;

    private double[] lastActInput;
    private double[] lastInput;

    private double[][] lastActInputs;
    private double[][] lastInputs;

    private double[] biases;
    private double[][] dweights; // gradients of weights needed if the optimizer is set.
    private double[] dbiases; //biases of layer.


    public FCL(int a, int b) {
        if (a < 1 || b < 1) {
            throw new IllegalArgumentException("Each layer must contain at least one neuron.");
        }

        weights = new double[a][b];

        RandomUtils.genTypeWeights(2, weights);

        this.useBiases = false;

        outputShape = new int[]{b};
        this.setInputShape(new int[]{a});

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


        for (int j = 0; j < weights[0].length; j++) {
            out[j] = act.definition(z[j]);
        }

        if (this.getNextLayer() != null) {
            this.getNextLayer().forward(out);
        } else {
            this.output = new Matrix(out);

        }

    }

    @Override
    public void forward(double[][] inputs) {

    }

    @Override
    public void forward(double[][][] input) {

    }

    @Override
    public void forward(double[][][][] inputs) {

    }

    @Override
    public void backward(double[] input, double learningRate) {
        this.learningRate = learningRate;
        this.getNextLayer().setLearningRate(learningRate);
        this.backward(input);

    }

    @Override
    public void backward(double[][] inputs, double learningRate) {

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

    @Override
    public void backward(double[] input) {
        double[] gradientOutput = new double[weights.length];

        double gradAct;
        double deltaWeight;
        double tmpW;

        for (int i = 0; i < weights.length; i++) {

            double gradientOutSum = 0;

            for (int j = 0; j < weights[0].length; j++) {

                gradAct = act.derivative(lastActInput[j]) * input[j];
                tmpW = weights[i][j];

                deltaWeight = gradAct * lastInput[i];

                weights[i][j] -= learningRate * deltaWeight;

                gradientOutSum += input[j] * gradAct * tmpW;
            }


            gradientOutput[i] = gradientOutSum;
        }

        if (useBiases) {

            for (int i = 0; i < biases.length; i++) {
                biases[i] -= learningRate * (lastActInput[i] * input[i]);
            }
        }


        if (this.getPreviousLayer() != null) {
            this.getPreviousLayer().backward(gradientOutput);
        }

    }

    @Override
    public void backward(double[][] inputs) {

    }

    @Override
    public void backward(double[][][] input) {

    }

    @Override
    public void backward(double[][][][] inputs) {

    }

    @Override
    public int[] getOutputShape() {
        return outputShape;
    }

    @Override
    public String export() {
        return null;
    }

    @Override
    public Matrix getOutput() {
        return output;
    }
}
