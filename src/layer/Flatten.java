package layer;

import load.writeUtils;
import utils.Array_utils;
import utils.Matrix;

import java.util.Arrays;

import static utils.Array_utils.getShape;

public class Flatten extends Layer {

    public Flatten(int[] inputShape) {

        this.inputShape = inputShape;
        this.outputShape = new int[]{Array_utils.getFlattenInputShape(inputShape)};

    }

    public Flatten(int[] inputShape, int outputSize) {
        this.inputShape = inputShape;
        this.outputShape = new int[]{outputSize};
    }


    public void forward(double[][][] input) {

        this.inputShape = getShape(input);
        double[] out = Array_utils.flatten(input);

        if (this.getNextLayer() != null) {
            this.getNextLayer().forward(new Matrix(out));
        } else {
            this.output = new Matrix(out);
        }

    }


    public void forward(double[][][][] inputs) {
        this.inputShape = getShape(inputs[0]);
        double[][] out = new double[inputs.length][];
        for (int i = 0; i < inputs.length; i++) {
            out[i] = Array_utils.flatten(inputs[i]);
        }

        if (this.getNextLayer() != null) {
            this.getNextLayer().forward(new Matrix(out));
        } else {
            this.output = new Matrix(out);
        }
    }


    public void backward(double[] input, double learningRate) {
        this.nextLayer.setLearningRate(learningRate);
        this.backward(input);
    }


    public void backward(double[][] inputs, double learningRate) {
        this.nextLayer.setLearningRate(learningRate);
        this.backward(inputs);
    }

    @Override
    public void forward(Matrix m) {
        int dim = m.getDim();

        if (dim == 3) {
            this.forward(m.getData3D());
        } else if (dim == 4) {
            this.forward(m.getData4D());
        } else {
            System.out.println("Flatten Got unsupported Dimension");
        }

    }

    @Override
    public void backward(Matrix m) {
        int dim = m.getDim();

        if (dim == 1) {
            this.backward(m.getData1D());
        } else if (dim == 2) {
            this.backward(m.getData2D());
        } else {
            System.out.println("Got unsupported Dimension");
        }
    }

    @Override
    public void backward(Matrix m, double learningRate) {
        if (this.previousLayer != null) {
            this.previousLayer.setLearningRate(learningRate);
        }
        int dim = m.getDim();
        if (dim == 1) {
            this.backward(m.getData1D());
        } else if (dim == 2) {
            this.backward(m.getData2D());
        } else {
            System.out.println("Got unsupported Dimension");
        }
    }

    @Override
    public Matrix getWeights() {
        return null;
    }

    @Override
    public void setWeights(Matrix m) {

    }


    public void backward(double[] input) {
        double[][][] out = Array_utils.reFlat(input, this.inputShape);


        if (this.getPreviousLayer() != null) {
            this.getPreviousLayer().backward(new Matrix(out));
        }
    }


    public void backward(double[][] inputs) {
        double[][][][] out = new double[inputs.length][][][];
        for (int i = 0; i < inputs.length; i++) {
            out[i] = Array_utils.reFlat(inputs[i], this.inputShape);
        }
        if (this.getPreviousLayer() != null) {
            this.getPreviousLayer().backward(new Matrix(out));
        }
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

    @Override
    public boolean isEqual(Layer other) {

        other = (Flatten) other;
        if (Arrays.equals(other.getInputShape(), this.inputShape)) {
            return true;
        }

        return false;


    }


    public boolean isEqual(Flatten other) {

        if (Arrays.equals(other.getInputShape(), this.inputShape)) {
            return true;
        }

        return false;


    }

}
