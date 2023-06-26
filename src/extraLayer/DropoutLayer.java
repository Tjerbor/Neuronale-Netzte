package extraLayer;

import main.Dropout;
import main.LayerNew;
import utils.Matrix;

import java.util.Arrays;

public class DropoutLayer extends LayerNew {


    Dropout dropout;

    public DropoutLayer(double percentage) {
        dropout = new Dropout(percentage);
    }

    @Override
    public void forward(double[] input) {

        input = dropout.forward(input);

        if (this.nextLayer != null) {
            this.nextLayer.forward(input);
        } else {
            this.output = new Matrix(input);
        }

    }

    @Override
    public void forward(double[][] inputs) {
        inputs = dropout.forward(inputs);

        if (this.nextLayer != null) {
            this.nextLayer.forward(inputs);
        } else {
            this.output = new Matrix(inputs);
        }

    }

    @Override
    public void forward(double[][][] input) {
        input = dropout.forward(input);

        if (this.nextLayer != null) {
            this.nextLayer.forward(input);
        } else {
            this.output = new Matrix(input);
        }
    }

    @Override
    public void forward(double[][][][] inputs) {
        inputs = dropout.forward(inputs);

        if (this.nextLayer != null) {
            this.nextLayer.forward(inputs);
        } else {
            this.output = new Matrix(inputs);
        }
    }

    @Override
    public void backward(double[] input, double learningRate) {
        this.backward(input);

    }

    @Override
    public void backward(double[][] inputs, double learningRate) {
        this.backward(inputs);
    }

    @Override
    public Matrix getWeights() {
        return null;
    }

    @Override
    public void setWeights(Matrix m) {

    }

    @Override
    public void backward(double[] input) {
        input = dropout.backward(input);

        if (this.previousLayer != null) {
            this.previousLayer.backward(input);
        }
    }

    @Override
    public void backward(double[][] inputs) {
        inputs = dropout.backward(inputs);

        if (this.previousLayer != null) {
            this.previousLayer.backward(inputs);
        }
    }

    @Override
    public void backward(double[][][] input) {
        input = dropout.backward(input);

        if (this.previousLayer != null) {
            this.previousLayer.backward(input);
        }
    }

    @Override
    public void backward(double[][][][] inputs) {
        inputs = dropout.backward(inputs);

        if (this.previousLayer != null) {
            this.previousLayer.backward(inputs);
        }
    }

    @Override
    public String export() {
        return "dropout;" + dropout.getRate();
    }

    @Override
    public void setTraining(boolean training) {
        this.training = training;
        this.dropout.setTraining(training);
    }

    public String summary() {
        return "Dropout inputSize: " + Arrays.toString(getInputShape())
                + " outputSize: " + Arrays.toString(getOutputShape())
                + " parameterSize: " + parameters() + "\n";
    }
}
