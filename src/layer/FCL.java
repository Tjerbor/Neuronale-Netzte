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

        lastActInputs = z;


        for (int bs = 0; bs < weights[0].length; bs++) {
            for (int i = 0; i < weights[0].length; i++) {
                out[bs][i] = act.definition(z[bs][i]);
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
    public void backward(Matrix m) {
        if (m.getDim() == 1) {
            this.backward(m.getData1D());
        } else if (m.getDim() == 2) {
            this.backward(m.getData2D());
        } else {
            throw new IllegalArgumentException("Expected flatten Array got Dim: " + m.getDim());
        }

    }

    @Override
    public void backward(Matrix m, double learningRate) {

        if (this.previousLayer != null) {
            previousLayer.setLearningRate(learningRate);
        }

        this.learningRate = learningRate;
        if (m.getDim() == 1) {
            this.backward(m.getData1D());
        } else if (m.getDim() == 2) {
            this.backward(m.getData2D());
        } else {
            throw new IllegalArgumentException("Expected flatten Array got Dim: " + m.getDim());
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


            gradientOutput[i] += gradientOutSum;
        }

        if (useBiases) {

            for (int i = 0; i < biases.length; i++) {
                biases[i] -= learningRate * (lastActInput[i] * input[i]);
            }
        }


        if (this.getPreviousLayer() != null) {
            this.getPreviousLayer().backward(new Matrix(gradientOutput));
        }

    }

    public void backward(double[][] inputs) {
        double[][] gradientOutput = new double[inputs.length][weights.length];

        double gradAct;
        double deltaWeight;
        double tmpW;

        for (int bs = 0; bs < inputs.length; bs++) {


            for (int i = 0; i < weights.length; i++) {

                double gradientOutSum = 0;

                for (int j = 0; j < weights[0].length; j++) {

                    gradAct = act.derivative(lastActInputs[bs][j]) * inputs[bs][j];
                    tmpW = weights[i][j];

                    deltaWeight = gradAct * lastInputs[bs][i];

                    weights[i][j] -= learningRate * deltaWeight;

                    gradientOutSum += inputs[bs][j] * gradAct * tmpW;
                }


                gradientOutput[bs][i] += gradientOutSum;
            }

        }

        if (useBiases) {

            for (int bs = 0; bs < inputs.length; bs++) {
                for (int i = 0; i < biases.length; i++) {
                    biases[i] -= learningRate * (lastActInputs[bs][i] * inputs[bs][i]);
                }
            }
        }


        if (this.getPreviousLayer() != null) {
            this.getPreviousLayer().backward(new Matrix(gradientOutput));
        }

    }


    @Override
    public String export() {

        String s = "fullyconnectedlayer;" + useBiases + ";" + weights.length + ";" + weights[0].length + ";" + act.toString() + ";" + isNull(dropout) + "\n";

        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                if (i == weights.length - 1 && j == weights[0].length - 1) {
                    s += weights[i][j];
                } else {
                    s += weights[i][j] + ";";
                }
            }
        }


        if (useBiases) {
            s += "\n";

            for (int i = 0; i < biases.length; i++) {

                if (i == biases.length - 1) {
                    s += biases[i];
                } else {
                    s += biases[i] + ";";
                }
            }

        }

        return s + "\n";

    }

    @Override
    public String summary() {
        return "FCL inputSize: " + Arrays.toString(getInputShape())
                + " outputSize: " + Arrays.toString(getOutputShape())
                + " parameter" + parameters() + "\n";
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
