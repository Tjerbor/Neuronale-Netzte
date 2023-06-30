package layer;

import function.Activation;
import function.TanH;
import utils.Array_utils;
import utils.Matrix;
import utils.RandomUtils;

import java.util.Arrays;

import static load.writeUtils.writeShape;
import static load.writeUtils.writeWeights;

public class FastLinearLayer extends Layer {

    Matrix backwardOutput;
    Activation act = new TanH();
    double learningRate = 0.4;
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

    public FastLinearLayer(int a, int b, double learningRate, Activation act) {
        this.learningRate = learningRate;

        weights = new double[a][b];
        RandomUtils.genTypeWeights(2, weights); //weights between -0.1 bis 0.1
        this.inputShape = new int[]{a};
        this.outputShape = new int[]{b};
        this.act = act;

    }

    public FastLinearLayer(int a, int b) {
        weights = new double[a][b];
        RandomUtils.genTypeWeights(2, weights); //weights between -0.1 bis 0.1
        this.inputShape = new int[]{a};
        this.outputShape = new int[]{b};

    }

    public FastLinearLayer(int a, int b, Activation act) {
        weights = new double[a][b];
        RandomUtils.genTypeWeights(2, weights); //weights between -0.1 bis 0.1
        this.inputShape = new int[]{a};
        this.outputShape = new int[]{b};
        this.act = act;

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
            this.nextLayer.forward(new Matrix(out));
        } else {
            this.output = new Matrix(out);
        }
    }


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
            this.nextLayer.forward(new Matrix(out));
        }

    }


    @Override
    public void forward(Matrix m) {

        if (m.getDim() == 2) {
            this.forward(m.getData2D());
        } else if (m.getDim() == 1) {
            this.forward(m.getData1D());
        } else {
            throw new IllegalArgumentException("Expected flattened Layer: " + m.getDim());
        }
    }

    @Override
    public void genWeights(int type) {
        RandomUtils.genTypeWeights(type, weights);

    }

    @Override
    public int parameters() {
        return weights.length * weights[0].length;
    }

    @Override
    public void backward(Matrix m) {
        if (m.getDim() == 2) {
            this.backward(m.getData2D());
        } else if (m.getDim() == 1) {
            this.backward(m.getData1D());
        }
    }

    @Override
    public void backward(Matrix m, double learningRate) {
        this.learningRate = learningRate;

        if (m.getDim() == 2) {
            this.backward(m.getData2D());
        } else if (m.getDim() == 1) {
            this.backward(m.getData1D());
        }
    }

    public Matrix getWeights() {
        Double[][] d;
        return new Matrix(weights);
        //Stream.of(boxed).mapToDouble(Double::doubleValue).toArray();

    }


    public void setWeights(double[][] weights) {

        System.out.println(weights.length + " " + weights[0].length);
        if (weights.length != weights.length || weights[0].length != weights[0].length) {
            throw new IllegalArgumentException("given weights are smaller.");
        }
        this.weights = weights;
    }

    @Override
    public void setWeights(Matrix m) {

        double[][] w = m.getData2D();
        System.out.println(w.length + " " + w[0].length);
        if (w.length != weights.length || weights[0].length != w[0].length) {
            throw new IllegalArgumentException("given weights are smaller.");
        }
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

        backwardOutput = new Matrix(dLdX);
        if (this.previousLayer != null) {
            this.previousLayer.backward(new Matrix(dLdX));
        }
    }

    public Matrix getBackwardOutput() {
        return this.backwardOutput;
    }

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
            this.previousLayer.backward(new Matrix(dInputs));
        }

    }

    @Override
    public String export() {

        String s = "fastlinearlayer;" + weights.length + ";" + weights[0].length + ";" + act.toString() + "\n";
        s += writeWeights(weights);

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

    @Override
    public boolean isEqual(Layer other) {

        FastLinearLayer other2 = (FastLinearLayer) other;

        if (Arrays.equals(other2.getInputShape(), this.inputShape) && this.getWeights().isEquals(other2.getWeights()) && this.act.isEquals(act)) {
            return true;
        }

        if (!(Arrays.equals(other2.getInputShape(), this.inputShape))) {
            System.out.println("inputShape was different: this: " + writeShape(inputShape) + " other: " + writeShape(other2.getInputShape()));
        }

        return false;


    }

    public boolean isEqual(FastLinearLayer other2) {

        if (Arrays.equals(other2.getInputShape(), this.inputShape) && this.getWeights().isEquals(other2.getWeights()) && this.act.isEquals(other2.act)) {
            return true;
        }

        if (!(Arrays.equals(other2.getInputShape(), this.inputShape))) {
            System.out.println("inputShape was different: this: " + writeShape(inputShape) + " other: " + writeShape(other2.getInputShape()));
        }


        return false;


    }


}
