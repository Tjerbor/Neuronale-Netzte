package layer;

import utils.Array_utils;
import utils.Utils;

import java.util.Arrays;

/**
 * normalize the data of a batch a cross that batch.
 * this is commonly used and older than the idea of Layer normalization.
 * often used with images and convolution layers.
 * braucht mehrere Input daten. Ein Beispiel kommt noch.
 */

public class BatchNorm1D {


    double epsilon = 1e-8;
    double momentum = 0.9;
    double[] gamma;
    double[] gammaGrad;
    double[] biases;
    double[] biasesGrad;

    boolean training = false;
    boolean useMomentum = false;

    double[] runningMean;
    double[] runningVar;


    double[] stddev;
    double[] var;
    double[][] standart_inputs;
    double[][] minus_mean;

    public BatchNorm1D(int input_size) {
        this.gamma = new double[input_size];
        this.biases = new double[input_size];
        this.runningMean = new double[input_size];
        this.runningVar = new double[input_size];
        Arrays.fill(gamma, 1);
        //Arrays.fill(biases, 0);
        //Arrays.fill(runningMean, 0);
        //Arrays.fill(runningVar, 0);


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

    public double[] sqrt_array(double[] a) {
        double[] out = new double[a.length];

        for (int i = 0; i < a.length; i++) {
            out[i] = Math.sqrt(a[i]);
        }
        return out;
    }


    public double[][] forward(double[][] inputs) {


        double[] mean = Utils.mean_axis_0(inputs);
        this.var = Array_utils.var_axis_0(inputs);


        System.out.println(var.length);

        if (useMomentum) {
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
        Utils.matmul2D_1D(out, gamma);
        Utils.addMatrix(out, biases);
        return out;
    }

    public double[][] backwardNew(double[][] grad_inputs) {

        int N = grad_inputs.length;
        double[] std = this.stddev;
        double[][] x_centered = minus_mean;
        double[][] x_norm = standart_inputs;

        double[] dgama = Array_utils.sum_axis_0(Utils.multiply(grad_inputs, x_norm));
        double[] dbeta = Array_utils.sum_axis_0(grad_inputs);


        double[][] dx_norm = Utils.matmul2D_1D(grad_inputs, gamma);
        double[][] dx_centered = Array_utils.Matrix2D_div1D(dx_norm, std);

        double[] dmean = Array_utils.addMatrixScalar(Array_utils.sum_axis_0(dx_centered), 2 / N);
        dmean = Array_utils.multiply1D(dmean, Array_utils.sum_axis_0(x_centered));

        double[][] dstdTmp = Array_utils.zerosLike(dx_norm);

        //(dx_norm * x_centered * -std**(-2))
        for (int i = 0; i < dx_norm.length; i++) {
            for (int j = 0; j < dx_norm[0].length; j++) {
                dstdTmp[i][j] = dx_norm[i][j] * x_centered[i][j] * -Math.pow(std[j], -2);
            }
        }


        double[] dstd = Array_utils.sum_axis_0(dstdTmp);
        //double[] dvar = dstd / 2 / std;
        double[] dvar = Array_utils.zerosLike(dstd);
        for (int i = 0; i < dstd.length; i++) {
            dvar[i] = dstd[i] / 2 / std[i];
        }

        double[][] dx = Array_utils.zerosLike(dx_centered);

        for (int i = 0; i < dx.length; i++) {
            for (int j = 0; j < dx[0].length; j++) {
                dx[i][j] = dx_centered[i][j] + (dmean[j] + dvar[j] * 2 * x_centered[i][j]
                ) / N;
            }
        }


        return dx;


    }


    public double[][] backward(double[][] grad_inputs) {

        int batch_size = grad_inputs.length;

        double[][] standard_grad = Utils.matmul2D_1D(grad_inputs, gamma);

        double[] stddev_inv = Array_utils.div_matrix_by_scalarRE(this.stddev, 1);

        double[] var_pow = new double[this.var.length];
        for (int i = 0; i < var.length; i++) {
            var_pow[i] = Math.pow(var[i], -3 / 2);
        }


        double[][] var_grad_tmp = new double[grad_inputs.length][grad_inputs[0].length];
        for (int i = 0; i < grad_inputs.length; i++) {
            for (int j = 0; j < grad_inputs[0].length; j++) {
                var_grad_tmp[i][j] = standard_grad[i][j] * this.minus_mean[i][j] * -0.5 * var_pow[j];
            }
        }

        double[] var_grad = Array_utils.sum_axis_0(var_grad_tmp);

        double[][] aux_x_minus_mean = new double[minus_mean.length][minus_mean[0].length];
        for (int i = 0; i < minus_mean.length; i++) {
            for (int j = 0; j < minus_mean[0].length; j++) {
                aux_x_minus_mean[i][j] = 2 * minus_mean[i][j] / batch_size;
            }
        }

        double[] neg_sum_aux = Array_utils.neg_sum_axis_0(aux_x_minus_mean);
        double[] mean_grad_ = Array_utils.multiply1D(var_grad, neg_sum_aux);


        double[][] mean_grad_part1 = new double[standard_grad.length][standard_grad[0].length];
        for (int i = 0; i < standard_grad.length; i++) {
            for (int j = 0; j < standard_grad[0].length; j++) {
                mean_grad_part1[i][j] = standard_grad[i][j] * (-stddev_inv[j]);


            }
        }

        double[] mean_grad = Array_utils.sum_axis_0(mean_grad_part1);
        Array_utils.addMatrix(mean_grad, mean_grad_);
        double[][] gammaGradTmp = Utils.multiply(grad_inputs, standart_inputs);


        biasesGrad = Array_utils.sum_axis_0(grad_inputs);
        gammaGrad = Array_utils.sum_axis_0(gammaGradTmp);


        double[][] out = new double[grad_inputs.length][grad_inputs[0].length];
        for (int i = 0; i < standard_grad.length; i++) {
            for (int j = 0; j < standard_grad[0].length; j++) {
                out[i][j] = standard_grad[i][j] * stddev_inv[j] + var_grad[j]
                        * aux_x_minus_mean[i][j] + mean_grad[j] / batch_size;
            }
        }
        return out;


    }

    public double[][] backward(double[][] grad_inputs, double learningRate) {

        double[][] out = this.backward(grad_inputs);
        this.updateParameter(learningRate);
        return out;

    }

    public void updateParameter(double learningRate) {
        // gamma -= gammaGrad * learningRate
        Utils.updateParameter(this.biases, this.biasesGrad, learningRate);
        Utils.updateParameter(this.gamma, this.gammaGrad, learningRate);


    }
}
