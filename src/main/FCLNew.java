package main;

import layer.Activation;
import layer.TanH;
import utils.Matrix;

import java.util.Arrays;
import java.util.Random;

public class FCLNew extends LayerNew {


    Activation act = new TanH();
    double learningRate;
    double[][] weights;
    double[] biases;
    private double[] lastActInput;

    private double[] lastInput;
    private double[][] lastInputs;
    private double[][] lastActInputs;

    public FCLNew(int _inLength, int _outLength, double learningRate) {
        this.learningRate = learningRate;

        weights = new double[_inLength][_outLength];
        setRandomWeights();
        this.inputShape = new int[]{_inLength};
        this.outputShape = new int[]{_outLength};

    }

    public void setRandomWeights() {
        Random random = new Random();

        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                weights[i][j] = random.nextDouble(-0.1, 0.1);
            }
        }
    }

    public void forward(double[] input) {

        lastInput = input;

        double[] tmp = new double[weights[0].length];
        double[] out = new double[weights[0].length];

        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                tmp[j] += input[i] * weights[i][j];
            }

        }

        if (useBiases) {
            for (int i = 0; i < biases.length; i++) {
                tmp[i] += biases[i];
            }
        }

        lastActInput = tmp;

        for (int j = 0; j < out.length; j++) {
            out[j] = act.definition(tmp[j]);
        }

        if (this.dropout != null) {
            out = this.dropout.forward(out);
        }

        if (this.nextLayer != null) {
            this.nextLayer.forward(out);
        } else {
            this.output = new Matrix(out);
        }
    }

    @Override
    public void forward(double[][] inputs) {

        lastInputs = inputs;


        double[][] tmp = new double[inputs.length][weights[0].length];
        double[][] out = new double[inputs.length][weights[0].length];

        for (int bs = 0; bs < inputs.length; bs++) {
            for (int i = 0; i < inputs[0].length; i++) {
                for (int j = 0; j < weights[0].length; j++) {
                    tmp[bs][j] += inputs[bs][i] * weights[i][j];
                }
            }
        }

        lastActInputs = tmp;

        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < out.length; j++) {
                out[i][j] = act.definition(tmp[i][j]);
            }
        }
        if (this.dropout != null) {
            out = this.dropout.forward(out);
        }

        if (this.nextLayer != null) {
            this.nextLayer.forward(out);
        } else {
            this.output = new Matrix(out);
        }
    }

    @Override
    public void forward(double[][][] input) {

    }

    @Override
    public void forward(double[][][][] inputs) {

    }

    @Override
    public void backward(double[] input, double learningRate) {

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


    public void backward(double[] gradInput) {


        double[] gradOutput = new double[weights.length];

        double dAct;
        double dIn;
        double dweight;


        for (int k = 0; k < weights.length; k++) {

            double dLdX_sum = 0;

            for (int j = 0; j < weights[0].length; j++) {

                dAct = act.derivative(lastActInput[j]);
                dIn = lastInput[k];

                if (this.dropout != null) {
                    dweight = dropout.backward(gradInput[j] * dIn * dAct, k, 1);
                } else {
                    dweight = gradInput[j] * dIn * dAct;
                }
                dLdX_sum += gradInput[j] * dAct * weights[k][j];
                weights[k][j] -= dweight * learningRate;
            }

            gradOutput[k] = dLdX_sum;
        }

        if (this.previousLayer != null) {
            this.previousLayer.backward(gradOutput);
        }
    }

    @Override
    public void backward(double[][] inputs) {

        double[][] gradOutput = new double[inputs.length][weights.length];

        double dAct;
        double dIn;
        double dweight;


        for (int bs = 0; bs < inputs.length; bs++) {

            for (int i = 0; i < weights.length; i++) {

                double dLdX_sum = 0;

                for (int j = 0; j < weights[0].length; j++) {

                    dAct = act.derivative(lastActInput[j]);
                    dIn = lastInput[i];

                    if (this.dropout != null) {
                        dweight = dropout.backward(inputs[bs][j] * dIn * dAct, bs, i);
                    } else {
                        dweight = inputs[bs][j] * dIn * dAct;
                    }


                    dLdX_sum += inputs[bs][j] * dAct * weights[i][j];
                    weights[i][j] -= dweight * learningRate;
                }

                gradOutput[bs][i] = dLdX_sum;
            }

        }

        if (this.previousLayer != null) {
            this.previousLayer.backward(gradOutput);
        }
    }

    @Override
    public void backward(double[][][] input) {

    }

    @Override
    public void backward(double[][][][] inputs) {

    }

    @Override
    public String export() {
        return null;
    }

    @Override
    public Matrix getOutput() {
        return output;
    }

    @Override
    public String summary() {
        return "FullyConnectedLayer inputSize: " + Arrays.toString(getInputShape())
                + " outputSize: " + Arrays.toString(getOutputShape())
                + " parameter" + parameters() + "\n";
    }


    @Override
    public boolean isEqual(LayerNew other2) {

        FCLNew other = (FCLNew) other2;

        if (other.getInputShape() == this.inputShape && this.getWeights() == other.getWeights() && this.act == other.act) {
            return true;
        }

        return false;
    }


}
