package extraLayer;

import main.LayerNew;
import utils.Array_utils;
import utils.Matrix;
import utils.Utils;

import java.util.Arrays;

public class BatchNorm extends LayerNew {


    double epsilon = 1e-5;
    double momentum = 0.1;
    double[] gamma;
    double[] gammaGrad;
    double[] biases;
    double[] biasesGrad;
    double[] runningMean;
    double[] runningVar;

    double[] stddev;
    double[] var;
    double[][] standart_inputs;
    double[][] minus_mean;
    double[][][][] minus_mean4D;

    double[][][][] standart_inputs4D;

    int inputSize;

    Matrix output;

    int[] inputShape;

    public BatchNorm(int input_size) {
        this.gamma = new double[input_size];
        this.biases = new double[input_size];
        this.runningMean = new double[input_size];
        this.runningVar = new double[input_size];
        Arrays.fill(gamma, 1);
        Arrays.fill(runningVar, 1);

    }

    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    public double[] sqrt_array(double[] a) {
        double[] out = new double[a.length];

        for (int i = 0; i < a.length; i++) {
            out[i] = Math.sqrt(a[i]);
        }
        return out;
    }


    public void backward(double[][][][] grad_inputs, double learningRate) {

        this.learningRate = learningRate;
        this.backward(grad_inputs);

    }

    @Override
    public void backward(double[][][][] grad_inputs) {
        int N = grad_inputs[0].length;

        double[] std = this.stddev;
        double[][][][] x_centered = minus_mean4D;
        double[][][][] x_norm = standart_inputs4D;

        double[] dgama = Array_utils.sum_axis_0_2_3(Utils.multiply(grad_inputs, x_norm));
        double[] dbeta = Array_utils.sum_axis_0_2_3(grad_inputs);

        double[][][][] dx_norm = Array_utils.multiply_axis1(grad_inputs, gamma);
        double[][][][] dx_centered = Array_utils.Matrix_div_axis_1(dx_norm, std);

        double[] dmean = Array_utils.addMatrixScalar(Array_utils.sum_axis_0_2_3(dx_centered), 2 / N);
        dmean = Utils.multiply(dmean, Array_utils.sum_axis_0_2_3(x_centered));

        double[][][][] dstdTmp = Array_utils.zerosLike(dx_norm);

        //(dx_norm * x_centered * -std**(-2))
        for (int i = 0; i < dx_norm.length; i++) {
            for (int j = 0; j < dx_norm[0].length; j++) {
                for (int k = 0; k < dx_norm[0][0].length; k++) {
                    for (int l = 0; l < dx_norm[0][0][0].length; l++) {
                        dstdTmp[i][j][k][l] = dx_norm[i][j][k][l] * x_centered[i][j][k][l] * -Math.pow(std[j], -2);
                    }

                }

            }
        }


        double[] dstd = Array_utils.sum_axis_0_2_3(dstdTmp);
        //double[] dvar = dstd / 2 / std;
        double[] dvar = Array_utils.zerosLike(dstd);
        for (int i = 0; i < dstd.length; i++) {
            dvar[i] = dstd[i] / 2 / std[i];

        }

        double[][][][] dx = Array_utils.zerosLike(dx_centered);

        for (int i = 0; i < dx.length; i++) {
            for (int j = 0; j < dx[0].length; j++) {
                for (int k = 0; k < dx[0][0].length; k++) {
                    for (int l = 0; l < dx[0][0][0].length; l++) {
                        dx[i][j][k][l] = dx_centered[i][j][k][l] + (dmean[j] + dvar[j] * 2 * x_centered[i][j][k][l]
                        ) / N;
                    }

                }

            }
        }

        this.updateParameter();
        if (this.getPreviousLayer() != null) {
            this.getPreviousLayer().backward(dx);
        }

    }

    @Override
    public int[] getOutputShape() {
        return this.inputShape;
    }

    @Override
    public String export() {
        return null;
    }

    @Override
    public Matrix getOutput() {
        if (output == null) {
            return null;
        }
        return output;
    }

    public void updateParameter() {
        // gamma -= gammaGrad * learningRate
        Utils.updateParameter(this.biases, this.biasesGrad, learningRate);
        Utils.updateParameter(this.gamma, this.gammaGrad, learningRate);


    }

    @Override
    public void forward(double[][][][] inputs) {


        double[] mean = Array_utils.mean_axis_0_2_3(inputs);
        this.var = Array_utils.var_axis_0_2_3(inputs);
        //normalize varianz.

        for (int i = 0; i < mean.length; i++) {
            runningMean[i] = momentum * runningMean[i] + (1 - momentum) * mean[i];
            runningVar[i] = momentum * runningVar[i] + (1 - momentum) * var[i];
        }

        for (int i = 0; i < var.length; i++) {
            var[i] += epsilon;
        }

        this.stddev = sqrt_array(var);

        this.minus_mean4D = new double[inputs.length][inputs[0].length][inputs[0][0].length][inputs[0][0][0].length];


        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                for (int k = 0; k < inputs[0][0].length; k++) {
                    for (int l = 0; l < inputs[0][0][0].length; l++) {
                        minus_mean4D[i][j][k][l] = (inputs[i][j][k][l] - mean[j]);
                    }

                }
            }
        }

        standart_inputs = minus_mean.clone();
        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                for (int k = 0; k < inputs[0][0].length; k++) {
                    for (int l = 0; l < inputs[0][0][0].length; l++) {
                        standart_inputs4D[i][j][k][l] /= stddev[j];
                    }

                }

            }
        }

        double[][][][] out = Array_utils.copyArray(this.standart_inputs4D);

        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                for (int k = 0; k < inputs[0][0].length; k++) {
                    for (int l = 0; l < inputs[0][0][0].length; l++) {

                        out[i][j][k][l] = out[i][j][k][l] * gamma[j] + biases[j];

                    }
                }
            }
        }

        this.output = new Matrix<>(out);
    }

    @Override
    public void backward(double[] input, double learningRate) {

    }

    @Override
    public void backward(double[][] inputs, double learningRate) {
        this.learningRate = learningRate;
        this.backward(inputs);
    }

    @Override
    public void forward(double[] input) {
        throw new IllegalArgumentException("Batch norm is meant to Normalize Batch.");
    }

    @Override
    public void forward(double[][] inputs) {


        double[] mean = Utils.mean_axis_0(inputs);
        this.var = Array_utils.var_axis_0(inputs);


        System.out.println(var.length);

        if (training) {
            for (int i = 0; i < mean.length; i++) {
                runningMean[i] = momentum * runningMean[i] + (1 - momentum) * mean[i];
                runningVar[i] = momentum * runningVar[i] + (1 - momentum) * var[i];
            }


        }

        //normalize Varianz.
        for (int i = 0; i < var.length; i++) {
            var[i] += epsilon;
        }

        this.stddev = sqrt_array(var);


        this.minus_mean = new double[inputs.length][inputs[0].length];


        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                minus_mean[i][j] = (inputs[i][j] - mean[j]);
            }
        }

        standart_inputs = Array_utils.zerosLike(minus_mean);

        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                standart_inputs[i][j] = minus_mean[i][j] / stddev[j];
            }
        }

        double[][] out = Array_utils.copyArray(this.standart_inputs);

        for (int i = 0; i < out.length; i++) {
            for (int j = 0; j < out[0].length; j++) {
                out[i][j] = out[i][j] * gamma[j] + biases[j];
            }
        }

        if (this.getNextLayer() != null) {
            this.getNextLayer().forward(out);
        } else {
            this.output = new Matrix(out);
        }
    }

    @Override
    public void forward(double[][][] input) {
        throw new RuntimeException("Not Implemented");
    }

    @Override
    public Matrix getWeights() {

        double[][] c = new double[][]{gamma, biases, runningVar, runningMean};
        return new Matrix(c);
    }

    @Override
    public void setWeights(Matrix m) {

        double[][] w = m.getData2D();

        if (w.length == 4) {
            gamma = w[0];
            biases = w[1];
            runningVar = w[2];
            runningMean = w[3];

        } else if (w.length == 2) {
            gamma = w[0];
            biases = w[1];
        } else {
            throw new IllegalArgumentException("Either accepts all 4 weights or gamma and biases(beta)");
        }


    }

    @Override
    public void backward(double[] input) {

    }

    @Override
    public void backward(double[][] inputs) {

    }

    @Override
    public void backward(double[][][] input) {

    }
}
