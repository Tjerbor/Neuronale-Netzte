package layer;


import main.LayerNew;
import utils.Array_utils;
import utils.Matrix;
import utils.Utils;

import java.util.Arrays;

import static load.writeUtils.writeShape;
import static utils.Array_utils.getShape;

/**
 * this Activation Layer is used for more flexibility for the Model Usage.
 */
public class ActivationLayer extends LayerNew {

    double[] input1D;
    double[][] inputs2D;
    double[][][] input3D;
    double[][][][] inputs4D;

    Activation act;


    public ActivationLayer(Activation act) {
        this.act = act;
    }

    public ActivationLayer(String act) {
        this.act = Utils.getActivation(act);
    }

    @Override
    public Matrix getOutput() {
        return output;
    }

    public void forward(double[][][] inputs) {
        this.input3D = Array_utils.copyArray(inputs);

        double[][][] c = Array_utils.zerosLike(inputs);
        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                for (int k = 0; k < inputs[0][0].length; k++) {
                    inputs[i][j][k] = act.definition(inputs[i][j][k]);
                }
            }
        }

        if (this.getNextLayer() != null) {
            this.getNextLayer().forward(c);
        } else {
            this.output = new Matrix(c);
        }


    }

    @Override
    public void backward(double[][][] gradInputs) {


        for (int i = 0; i < input3D.length; i++) {
            for (int j = 0; j < input3D[0].length; j++) {
                for (int k = 0; k < input3D[0][0].length; k++) {
                    gradInputs[i][j][k] = act.derivative(this.input3D[i][j][k]) * gradInputs[i][j][k];
                }
            }
        }
        if (this.getPreviousLayer() != null) {
            this.getPreviousLayer().backward(gradInputs);
        }


    }

    public void forward(double[][][][] inputs) {
        this.inputs4D = Array_utils.copyArray(inputs);

        double[][][][] c = Array_utils.zerosLike(inputs);
        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                for (int k = 0; k < inputs[0][0].length; k++) {
                    for (int l = 0; l < inputs[0][0][0].length; l++) {
                        c[i][j][k][l] = act.definition(inputs[i][j][k][l]);
                    }

                }
            }
        }

        if (this.getNextLayer() != null) {
            this.getNextLayer().forward(c);
        } else {
            this.output = new Matrix(c);
        }
    }


    public void backward(double[][][][] gradInputs) {
        for (int i = 0; i < inputs4D.length; i++) {
            for (int j = 0; j < inputs4D[0].length; j++) {
                for (int k = 0; k < inputs4D[0][0].length; k++) {
                    for (int l = 0; l < inputs4D[0][0][0].length; l++) {
                        gradInputs[i][j][k][l] = act.derivative(this.inputs4D[i][j][k][l]) * gradInputs[i][j][k][l];
                    }

                }
            }
        }

        if (this.getPreviousLayer() != null) {
            this.getPreviousLayer().backward(gradInputs);
        }

    }

    public void backward(double[][][][] gradInputs, double learningRate) {
        this.backward(gradInputs);
    }


    @Override
    public void forward(double[] input) {
        this.input1D = Array_utils.copyArray(input);

        double[] c = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            c[i] = act.definition(input[i]);
        }
        if (this.getNextLayer() != null) {
            this.getNextLayer().forward(c);
        } else {
            this.output = new Matrix(c);
        }
    }

    @Override
    public void forward(double[][] inputs) {
        this.inputs2D = Array_utils.copyArray(inputs);

        double[][] c = new double[inputs.length][inputs[0].length];
        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                c[i][j] = act.definition(inputs[i][j]);
            }

        }

        if (this.getNextLayer() != null) {
            this.getNextLayer().forward(c);
        } else {
            this.output = new Matrix(c);
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
        for (int i = 0; i < input.length; i++) {
            input[i] = act.derivative(this.input1D[i]) * input[i];
        }
        if (this.getPreviousLayer() != null) {
            this.getPreviousLayer().backward(input);
        }
    }

    @Override
    public void backward(double[][] inputs) {

        for (int i = 0; i < inputs2D.length; i++) {
            for (int j = 0; j < inputs2D[0].length; j++) {
                inputs[i][j] = act.derivative(this.inputs2D[i][j]) * inputs[i][j];
            }

        }
        if (this.getPreviousLayer() != null) {
            this.getPreviousLayer().backward(inputs);
        }
    }

    @Override
    public int parameters() {
        return 0;
    }

    @Override
    public int[] getInputShape() {

        if (input1D != null) {
            return getShape(input1D);
        } else if (inputs2D != null) {
            return getShape(inputs2D);
        } else if (input3D != null) {
            return getShape(input3D);
        } else if (inputs4D != null) {
            return getShape(inputs4D);
        } else {
            return null;
        }
    }

    @Override
    public int[] getOutputShape() {
        return this.getInputShape();
    }

    @Override
    public String export() {
        return "activation;" + act.toString().toLowerCase() + ";" + writeShape(inputShape);
    }

    @Override
    public String summary() {
        return act.toString() + " inputSize: " + Arrays.toString(getInputShape())
                + " outputSize: " + Arrays.toString(getOutputShape())
                + " parameter" + parameters() + "\n";
    }


    @Override
    public boolean isEqual(LayerNew other2) {

        ActivationLayer other = (ActivationLayer) other2;

        if (Arrays.equals(other.getInputShape(), this.inputShape) && this.act.isEquals(act)) {
            return true;
        }

        return false;
    }

}
