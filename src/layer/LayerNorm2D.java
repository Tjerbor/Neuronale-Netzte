package layer;

import utils.Array_utils;
import utils.Utils;

import java.util.Arrays;

import static utils.Array_utils.getShape;

public class LayerNorm2D {

    double epsilon = 1e-8;

    double momentum = 0.9;
    double[][] gamma;
    double[][] gammaGrad;
    double[][] biases;
    double[][] biasesGrad;


    double[][] stddev;
    double[][] var;
    double[][][] standart_inputs;
    double[][][] minus_mean;

    boolean training = false;
    boolean useMomentum = false;

    double[][] runningMean;
    double[][] runningVar;

    public LayerNorm2D(int inputSize, int inputSize2) {
        this.gamma = new double[inputSize][inputSize2];
        this.biases = new double[inputSize][inputSize2];
        this.runningMean = new double[inputSize][inputSize2];
        this.runningVar = new double[inputSize][inputSize2];

        gamma = Array_utils.onesLike(inputSize, inputSize2);


    }

    public void setEpsilon(double e) {
        epsilon = e;
    }

    public void setUseMomentum(boolean b) {
        useMomentum = b;
    }

    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }


    public void setTraining(boolean training) {
        this.training = training;
        if (!training) {
            this.useMomentum = false;
        }
    }

    public double[][] sqrt_array(double[][] a) {
        double[][] out = new double[a.length][a[0].length];

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                out[i][j] = Math.sqrt(a[i][j]);
            }

        }
        return out;
    }


    public double[][][] forward(double[][][] inputs) {


        double[][] mean = Array_utils.mean_axis_1(inputs);
        this.var = Array_utils.var_axis_1(inputs);
        //normalize varianz.


        if (useMomentum) {
            for (int i = 0; i < mean.length; i++) {
                for (int j = 0; j < mean[0].length; j++) {
                    runningMean[i][j] = momentum * runningMean[i][j] + (1 - momentum) * mean[i][j];
                    runningVar[i][j] = momentum * runningVar[i][j] + (1 - momentum) * var[i][j];
                }

            }


        }

        for (int i = 0; i < var.length; i++) {
            for (int j = 0; j < var[0].length; j++) {
                var[i][j] += epsilon;
            }

        }

        this.stddev = sqrt_array(var);


        this.minus_mean = Array_utils.zerosLike(inputs);


        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                for (int k = 0; k < inputs[0][0].length; k++) {
                    minus_mean[i][j][k] = (inputs[i][j][k] - mean[i][k]);
                }
            }
        }

        standart_inputs = minus_mean.clone();
        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                for (int k = 0; k < inputs[0][0].length; k++) {
                    standart_inputs[i][j][k] /= stddev[j][k];
                }

            }
        }

        double[][][] out = this.standart_inputs.clone();
        Utils.matmul3D_2D(out, gamma);
        System.out.println(Arrays.toString(getShape(out)));
        Array_utils.addMatrix(out, biases);
        return out;

    }

    public double[][][] backward(double[][][] grad_inputs) {

        int N = grad_inputs.length;
        double[][] std = this.stddev;
        double[][][] x_centered = minus_mean;
        double[][][] x_norm = standart_inputs;

        double[][] dgama = Array_utils.sum_axis_1(Utils.multiply(grad_inputs, x_norm));
        double[][] dbeta = Array_utils.sum_axis_1(grad_inputs);


        double[][][] dx_norm = Utils.multiply(grad_inputs, gamma);
        double[][][] dx_centered = Array_utils.Matrix3Ddiv2D(dx_norm, std);

        double[][] dmean = Array_utils.addMatrixScalar(Array_utils.sum_axis_0(dx_centered), 2 / N);
        dmean = Utils.multiply(dmean, Array_utils.sum_axis_1(x_centered));

        double[][][] dstdTmp = Array_utils.zerosLike(dx_norm);

        //(dx_norm * x_centered * -std**(-2))
        for (int i = 0; i < dx_norm.length; i++) {
            for (int j = 0; j < dx_norm[0].length; j++) {
                for (int k = 0; k < dx_norm[0][0].length; k++) {
                    dstdTmp[i][j][k] = dx_norm[i][j][k] * x_centered[i][j][k] * -Math.pow(std[i][k], -2);
                }

            }
        }


        double[][] dstd = Array_utils.sum_axis_1(dstdTmp);
        //double[] dvar = dstd / 2 / std;
        double[][] dvar = Array_utils.zerosLike(dstd);
        for (int i = 0; i < dstd.length; i++) {
            for (int j = 0; j < dstd[0].length; j++) {
                dvar[i][j] = dstd[i][j] / 2 / std[i][j];
            }

        }
        double[][][] dx = Array_utils.zerosLike(dx_centered);

        for (int i = 0; i < dx.length; i++) {
            for (int j = 0; j < dx[0].length; j++) {
                for (int k = 0; k < dx[0][0].length; k++) {
                    dx[i][j][k] = dx_centered[i][j][k] + (dmean[i][k] + dvar[i][k] * 2 * x_centered[i][j][k]
                    ) / N;
                }

            }
        }
        return dx;


    }

    public double[][][] backward(double[][][] grad_inputs, double learningRate) {

        double[][][] out = this.backward(grad_inputs);
        this.updateParameter(learningRate);
        return out;

    }

    public void updateParameter(double learningRate) {
        // gamma -= gammaGrad * learningRate
        Utils.updateParameter(this.biases, this.biasesGrad, learningRate);
        Utils.updateParameter(this.gamma, this.gammaGrad, learningRate);
    }

}
