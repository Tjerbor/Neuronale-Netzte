package extraLayer;

import load.writeUtils;
import main.LayerNew;
import utils.Array_utils;
import utils.Matrix;

import java.util.Arrays;

import static utils.Array_utils.getShape;

public class Flatten extends LayerNew {

    public Flatten(int[] inputShape) {

        this.inputShape = inputShape;
        this.outputShape = new int[]{Array_utils.getFlattenInputShape(inputShape)};

    }

    public Flatten(int[] inputShape, int outputSize) {
        this.inputShape = inputShape;
        this.outputShape = new int[]{outputSize};
    }

    @Override
    public void forward(double[] input) {
        throw new IllegalArgumentException("Array is already flat.");

    }

    @Override
    public void forward(double[][] inputs) {
        throw new IllegalArgumentException("Array is already flat.");
    }

    @Override
    public void forward(double[][][] input) {

        this.inputShape = getShape(input);
        double[] out = Array_utils.flatten(input);

        if (this.getNextLayer() != null) {
            this.getNextLayer().forward(input);
        } else {
            this.output = new Matrix(out);
        }

    }

    @Override
    public void forward(double[][][][] inputs) {
        this.inputShape = getShape(inputs[0]);
        double[][] out = new double[inputs.length][];
        for (int i = 0; i < inputs.length; i++) {
            out[i] = Array_utils.flatten(inputs[i]);
        }

        if (this.getNextLayer() != null) {
            this.getNextLayer().forward(inputs);
        } else {
            this.output = new Matrix(out);
        }
    }

    @Override
    public void backward(double[] input, double learningRate) {
        this.nextLayer.setLearningRate(learningRate);
        this.backward(input);
    }

    @Override
    public void backward(double[][] inputs, double learningRate) {
        this.nextLayer.setLearningRate(learningRate);
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
        double[][][] out = Array_utils.reFlat(input, this.inputShape);
        ;

        if (this.getPreviousLayer() != null) {
            this.getPreviousLayer().backward(out);
        }
    }

    @Override
    public void backward(double[][] inputs) {
        double[][][][] out = new double[inputs.length][][][];
        for (int i = 0; i < inputs.length; i++) {
            out[i] = Array_utils.reFlat(inputs[i], this.inputShape);
        }
        if (this.getPreviousLayer() != null) {
            this.getPreviousLayer().backward(out);
        }
    }

    @Override
    public void backward(double[][][] input) {
        throw new IllegalArgumentException("Not supported to reFlat array with Dimension 3");
    }

    @Override
    public void backward(double[][][][] inputs) {
        throw new IllegalArgumentException("Not supported to reFlat array with Dimension 4");
    }

    @Override
    public String export() {
        return "flatten;" + writeUtils.writeShape(inputShape);
    }

    @Override
    public String summary() {
        return "Flatten inputSize: " + Arrays.toString(getInputShape())
                + " outputSize: " + Arrays.toString(getOutputShape())
                + " parameterSize: " + parameters() + "\n";
    }
}
