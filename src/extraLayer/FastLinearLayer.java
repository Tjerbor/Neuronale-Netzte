package extraLayer;

import layer.Activation;
import layer.TanH;
import main.LayerNew;
import utils.Array_utils;
import utils.Matrix;
import utils.RandomUtils;

public class FastLinearLayer extends LayerNew {


    Activation act = new TanH();
    double learningRate;
    double[][] weights;
    private double[] lastZ;
    private double[][] lastZs;
    private double[] lastX;
    private double[][] lastXs;

    public FastLinearLayer(int a, int b, double learningRate) {
        this.learningRate = learningRate;

        weights = new double[a][b];
        RandomUtils.genTypeWeights(2, weights); //weights between -0.1 bis 0.1
        this.inputShape = new int[]{a};
        this.outputShape = new int[]{b};

    }

    public FastLinearLayer(int a, int b) {
        weights = new double[a][b];
        RandomUtils.genTypeWeights(2, weights); //weights between -0.1 bis 0.1
        this.inputShape = new int[]{a};
        this.outputShape = new int[]{b};

    }

    public void forward(double[] input) {

        lastX = input;

        double[] z = new double[weights[0].length];
        double[] out = new double[weights[0].length];

        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                z[j] += input[i] * weights[i][j];
            }
        }

        lastZ = z;

        for (int j = 0; j < out.length; j++) {
            out[j] = act.definition(z[j]);
        }

        if (this.nextLayer != null) {
            this.nextLayer.forward(out);
        } else {
            this.output = new Matrix(out);
        }
    }

    @Override
    public void forward(double[][] inputs) {
        lastXs = inputs;

        double[][] z = new double[inputs.length][weights[0].length];
        double[][] out = new double[inputs.length][weights[0].length];

        for (int bs = 0; bs < inputs.length; bs++) {


            for (int i = 0; i < inputs[0].length; i++) {
                for (int j = 0; j < weights[0].length; j++) {
                    z[bs][j] += inputs[bs][i] * weights[i][j];
                }

            }

        }
        lastZs = z;
        for (int i = 0; i < out.length; i++) {
            for (int j = 0; j < out[0].length; j++) {
                out[i][j] = act.definition(z[i][j]);
            }
        }

        this.output = new Matrix(out);
        if (this.nextLayer != null) {
            this.nextLayer.forward(out);
        } else {
            this.output = new Matrix(out);
        }

    }

    @Override
    public void forward(double[][][] input) {
        throw new IllegalArgumentException("Expected shape 1Dim got 3Dim");
    }

    @Override
    public void forward(double[][][][] inputs) {
        throw new IllegalArgumentException("Expected shape 2Dim got 4Dim");
    }

    @Override
    public void backward(double[] input, double learningRate) {
        this.learningRate = learningRate;
        this.backward(input);
    }

    @Override
    public void backward(double[][] inputs, double learningRate) {
        this.learningRate = learningRate;
        this.backward(inputs);
    }

    public Matrix getWeights() {
        Double[][] d;
        return new Matrix(weights);
        //Stream.of(boxed).mapToDouble(Double::doubleValue).toArray();

    }

    @Override
    public void setWeights(Matrix m) {
        this.weights = m.getData2D();
    }

    public void backward(double[] dLdO) {

        double[] dLdX = new double[weights.length];

        double dOdz;
        double dzdw;
        double dLdw;
        double dzdx;

        for (int k = 0; k < weights.length; k++) {

            double dLdX_sum = 0;

            for (int j = 0; j < weights[0].length; j++) {

                dOdz = act.derivative(lastZ[j]);
                dzdw = lastX[k];
                dzdx = weights[k][j];

                dLdw = dLdO[j] * dOdz * dzdw;

                weights[k][j] -= dLdw * learningRate;

                dLdX_sum += dLdO[j] * dOdz * dzdx;
            }

            dLdX[k] = dLdX_sum;
        }

        if (this.previousLayer != null) {
            this.previousLayer.backward(dLdX);
        }
    }

    @Override
    public void backward(double[][] inputs) {


        double[][] dInputs = new double[inputs.length][weights.length];

        double dAct;
        double dweight;
        double[][] w = Array_utils.copyArray(weights);


        for (int bs = 0; bs < inputs.length; bs++) {

            for (int k = 0; k < weights.length; k++) {

                double dLdX_sum = 0;

                for (int j = 0; j < weights[0].length; j++) {

                    dAct = act.derivative(lastZs[bs][j]);

                    dweight = inputs[bs][j] * dAct * lastXs[bs][k];

                    weights[k][j] -= dweight * learningRate;

                    dLdX_sum += inputs[bs][j] * dAct * w[k][j];
                }

                dInputs[bs][k] = dLdX_sum;
            }


        }
        if (this.previousLayer != null) {
            this.previousLayer.backward(dInputs);
        }

    }

    @Override
    public void backward(double[][][] input) {
        throw new IllegalArgumentException("Expected shape 1Dim got 3Dim");
    }

    @Override
    public void backward(double[][][][] inputs) {
        throw new IllegalArgumentException("Expected shape 2Dim got 4Dim");
    }

    @Override
    public String export() {

        String s = "fastlinearlayer;" + weights.length + ";" + weights[0].length + ";" + act.toString() + ";" + "\n";


        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                if (i == weights.length - 1 && j == weights[0].length - 1) {
                    s += weights[i][j] + ";";
                } else {
                    s += weights[i][j];
                }
            }
        }

        return s;
    }

    @Override
    public Matrix getOutput() {
        return output;
    }

    @Override
    public String summary() {
        return "FastLinear inputSize: " + getInputShape()[0]
                + " outputSize: " + getOutputShape()[0]
                + " parameter: " + parameters() + "\n";
    }


}
