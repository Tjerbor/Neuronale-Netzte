package layer;

import function.Dropout;
import utils.Matrix;

import java.util.Arrays;

import static load.writeUtils.writeShape;

public class DropoutLayer extends Layer {


    Dropout dropout;

    Matrix backwardOutput; //for testing

    public DropoutLayer(double percentage) {
        dropout = new Dropout(percentage);
    }


    @Override
    public void forward(Matrix m) {
        int dim = m.getDim();
        if (dim == 1) {
            this.forward(m.getData1D());
        } else if (dim == 2) {
            this.forward(m.getData2D());
        } else if (dim == 3) {
            this.forward(m.getData3D());
        } else if (dim == 4) {
            this.forward(m.getData4D());
        } else {
            System.out.println("Got unsupported Dimension");
        }


    }

    @Override
    public void backward(Matrix m) {
        int dim = m.getDim();
        if (dim == 1) {
            this.backward(m.getData1D());
        } else if (dim == 2) {
            this.backward(m.getData2D());
        } else if (dim == 3) {
            this.backward(m.getData3D());
        } else if (dim == 4) {
            this.backward(m.getData4D());
        } else {
            System.out.println("Got unsupported Dimension");
        }
    }

    @Override
    public void backward(Matrix m, double learningRate) {
        if (this.previousLayer != null) {
            this.previousLayer.setLearningRate(learningRate);
        }
        this.backward(m);
    }


    public void forward(double[] input) {

        input = dropout.forward(input);

        if (this.nextLayer != null) {
            this.nextLayer.forward(new Matrix(input));
        } else {
            this.output = new Matrix(input);
        }

    }


    public void forward(double[][] inputs) {
        inputs = dropout.forward(inputs);

        if (this.nextLayer != null) {
            this.nextLayer.forward(new Matrix(inputs));
        } else {
            this.output = new Matrix(inputs);
        }

    }


    public void forward(double[][][] input) {
        input = dropout.forward(input);

        if (this.nextLayer != null) {
            this.nextLayer.forward(new Matrix(input));
        } else {
            this.output = new Matrix(input);
        }
    }


    public void forward(double[][][][] inputs) {
        inputs = dropout.forward(inputs);

        if (this.nextLayer != null) {
            this.nextLayer.forward(new Matrix(inputs));
        } else {
            this.output = new Matrix(inputs);
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
        input = dropout.backward(input);

        backwardOutput = new Matrix(input);
        if (this.previousLayer != null) {
            this.previousLayer.backward(new Matrix(input));
        }
    }


    public void backward(double[][] inputs) {
        inputs = dropout.backward(inputs);
        backwardOutput = new Matrix(inputs);
        if (this.previousLayer != null) {
            this.previousLayer.backward(new Matrix(inputs));
        }
    }


    public void backward(double[][][] input) {
        input = dropout.backward(input);
        backwardOutput = new Matrix(input);
        if (this.previousLayer != null) {
            this.previousLayer.backward(new Matrix(input));
        }
    }


    public void backward(double[][][][] inputs) {
        inputs = dropout.backward(inputs);

        backwardOutput = new Matrix(inputs);
        if (this.previousLayer != null) {
            this.previousLayer.backward(new Matrix(inputs));
        }
    }

    public Matrix getBackwardOutput() {
        return backwardOutput;
    }


    @Override
    public void setInputShape(int[] inputShape) {
        super.setInputShape(inputShape);
        this.outputShape = this.inputShape;
    }

    @Override
    public String export() {
        return "dropout;" + dropout.getRate() + ";" + writeShape(inputShape);
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

    @Override
    public boolean isEqual(Layer other) {

        DropoutLayer other2 = (DropoutLayer) other;
        if (Arrays.equals(other2.getInputShape(), this.inputShape) && this.dropout.getRate() == other2.dropout.getRate()) {
            return true;
        }

        if (!(Arrays.equals(other.getInputShape(), this.inputShape))) {
            System.out.println("this: " + Arrays.toString(inputShape) + " other: " + Arrays.toString(other.getInputShape()));
        }

        if (!(this.dropout.getRate() == other2.dropout.getRate())) {
            System.out.println("this: " + this.dropout.getRate() + " other: " + other2.dropout.getRate());
        }


        return false;


    }

    public boolean isEqual(DropoutLayer other) {

        if (Arrays.equals(other.getInputShape(), this.inputShape) && this.dropout.getRate() == other.dropout.getRate()) {
            return true;
        }

        if ((Arrays.equals(other.getInputShape(), this.inputShape))) {
            System.out.println("this: " + writeShape(inputShape) + " other: " + writeShape(getInputShape()));
        }

        if ((this.dropout.getRate() == other.dropout.getRate())) {
            System.out.println("this: " + this.dropout.getRate() + " other: " + other.dropout.getRate());
        }

        return false;


    }
}
