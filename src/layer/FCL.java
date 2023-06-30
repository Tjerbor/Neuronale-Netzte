package layer;

import optimizer.Optimizer;
import utils.Matrix;
import utils.RandomUtils;

import java.util.Arrays;

/**
 * naked version used for the builder because all other Layer
 * kann not be set in her.(dropout, activation )
 * needed because otherwise function are not rightfully used.
 * because the fullyConnected layer is speed Optimized.
 */

public class FCL extends Layer {

    Optimizer optimizer; //can be set in the layer so that every Optimizer can save one previous deltaWeights.
    Matrix backwardOutput;
    private double[][] weights;
    private double[] lastInput;
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

    public static String isNull(Object o) {

        if (o == null) {
            return "";
        } else {
            return "true";
        }
    }

    public void forward(double[][] inputs) {
        lastInputs = inputs;

        double[][] z = new double[inputs.length][weights[0].length];
        double[][] out = new double[inputs.length][weights[0].length];

        for (int bs = 0; bs < inputs.length; bs++) {

            for (int j = 0; j < weights[0].length; j++) {
                for (int i = 0; i < weights.length; i++) {

                    z[bs][j] += inputs[bs][i] * weights[i][j];
                }

                if (useBiases) {
                    z[bs][j] += biases[j];
                }
            }

        }

        if (this.getNextLayer() != null) {
            this.getNextLayer().forward(new Matrix(out));
        } else {
            this.output = new Matrix(out);

        }

    }

    public void forward(double[] input) {
        lastInput = input;


        double[] out = new double[weights[0].length];

        for (int j = 0; j < weights[0].length; j++) {
            for (int i = 0; i < weights.length; i++) {

                out[j] += input[i] * weights[i][j];
            }

            if (useBiases) {
                out[j] += biases[j];
            }
        }


        if (this.getNextLayer() != null) {
            this.getNextLayer().forward(new Matrix(out));
        } else {
            this.output = new Matrix(out);

        }

    }


    public void backward(double[] input, double learningRate) {
        this.learningRate = learningRate;

        if (this.previousLayer != null) {
            this.previousLayer.setLearningRate(learningRate);
        }

        this.backward(input);

    }

    public Matrix getBackwardOutput() {
        return this.backwardOutput;
    }


    @Override
    public void forward(Matrix m) {
        if (m.getDim() == 1) {
            this.forward(m.getData1D());
        } else if (m.getDim() == 2) {
            this.forward(m.getData2D());
        } else {
            throw new IllegalArgumentException("Expected flatten Array got Dim: " + m.getDim());
        }
    }

    @Override
    public void setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
    }

    @Override
    public void backward(Matrix m) {
        Matrix out;

        if (m.getDim() == 1) {
            if (optimizer == null) {
                out = this.backward(m.getData1D());
            } else {
                out = this.backwardOptimizer(m.getData1D());
            }

        } else if (m.getDim() == 2) {
            out = this.backward(m.getData2D());
        } else {
            throw new IllegalArgumentException("Expected flatten Array got Dim: " + m.getDim());
        }

        this.backwardOutput = out;
        if (this.previousLayer != null) {
            this.previousLayer.setIterationAt(iterationAt);
            this.previousLayer.backward(out);
        }

    }

    @Override
    public void backward(Matrix m, double learningRate) {


        Matrix out;
        this.learningRate = learningRate;
        if (m.getDim() == 1) {
            if (optimizer == null) {
                out = this.backward(m.getData1D());
            } else {
                out = this.backwardOptimizer(m.getData1D());
            }
        } else if (m.getDim() == 2) {
            out = this.backward(m.getData2D());
        } else {
            throw new IllegalArgumentException("Expected flatten Array got Dim: " + m.getDim());
        }
        if (this.previousLayer != null) {
            this.previousLayer.backward(out, learningRate);
        }
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
    public int parameters() {

        if (useBiases) {
            return weights.length * weights[0].length + weights[0].length;
        }
        return weights.length * weights[0].length;

    }

    @Override
    public void genWeights(int type) {
        if (this.useBiases) {
            RandomUtils.genTypeWeights(type, biases);
        }
        RandomUtils.genTypeWeights(type, weights);


    }

    @Override
    public void setUseBiases(boolean useBiases) {
        this.useBiases = useBiases;
        if (useBiases) {
            this.biases = new double[weights[0].length];
        }

    }

    public Matrix backward(double[] gradInput) {
        double[] dLdX = new double[weights.length];

        double dOdz;
        double dzdw;
        double dLdw;
        double dzdx;

        for (int k = 0; k < weights.length; k++) {

            double dLdX_sum = 0;

            for (int j = 0; j < weights[0].length; j++) {

                dzdw = lastInput[k];
                dzdx = weights[k][j];

                dLdw = gradInput[j] * 1 * dzdw;

                weights[k][j] -= dLdw * learningRate;

                dLdX_sum += gradInput[j] * 1 * dzdx;
            }

            dLdX[k] = dLdX_sum;
        }

        return new Matrix(dLdX);
    }

    public Matrix backwardOptimizer(double[] gradInput) {
        double[] dLdX = new double[weights.length];

        double[][] dWeights = new double[weights.length][weights[0].length];

        for (int k = 0; k < weights.length; k++) {

            double dLdX_sum = 0;

            for (int j = 0; j < weights[0].length; j++) {
                dWeights[k][j] += gradInput[j] * lastInput[k];
                dLdX_sum += gradInput[j] * weights[k][j];

            }

            dLdX[k] = dLdX_sum;
        }

        optimizer.setIterationAt(iterationAt);
        optimizer.updateParameter(weights, dWeights);
        if (useBiases) {
            optimizer.updateParameter(biases, gradInput);
        }

        return new Matrix(dLdX);
    }

    public Matrix backward(double[][] gradInputs) {
        double[][] gradOutput = new double[gradInputs.length][weights.length];


        for (int i = 0; i < gradInputs.length; i++) {
            this.lastInput = lastInputs[i];
            gradOutput[i] = this.backward(gradInputs[i]).getData1D();
        }

        return new Matrix(gradOutput);
    }


    @Override
    public String export() {

        StringBuilder s = new StringBuilder("fcl;" + useBiases + ";" + weights.length + ";" + weights[0].length + "\n");

        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                if (i == weights.length - 1 && j == weights[0].length - 1) {
                    s.append(weights[i][j]);
                } else {
                    s.append(weights[i][j]).append(";");
                }
            }
        }


        if (useBiases) {
            s.append("\n");

            for (int i = 0; i < biases.length; i++) {

                if (i == biases.length - 1) {
                    s.append(biases[i]);
                } else {
                    s.append(biases[i]).append(";");
                }
            }

        }

        return s.toString();

    }

    @Override
    public String summary() {
        return "FCL inputSize: " + Arrays.toString(getInputShape())
                + " outputSize: " + Arrays.toString(getOutputShape())
                + " parameters: " + parameters() + "\n";
    }

    @Override
    public boolean isEqual(Layer other2) {

        FCL other = (FCL) other2;

        if (other.getInputShape() == this.inputShape && this.getWeights() == other.getWeights() && this.act == other.act) {
            return true;
        }

        return false;
    }


}
