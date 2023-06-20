package layer;

import utils.Array_utils;
import utils.Utils;

import java.util.Arrays;

import static utils.Array_utils.zerosLike;

public class BatchNorm2D {

    double epsilon = 1e-5;
    double momentum = 0.1;
    double[] gamma;
    double[] gammaGrad;
    double[] biases;
    double[] biasesGrad;
    double[] runningMean;
    double[] runningVar;

    boolean training = true;
    double[] stddev;
    double[] var;
    double[][][][] standart_inputs;
    double[][][][] minus_mean;

    public BatchNorm2D(int inputSize) {
        this.gamma = new double[inputSize];
        this.biases = new double[inputSize];
        this.runningMean = new double[inputSize];
        this.runningVar = new double[inputSize];
        Arrays.fill(gamma, 1.0);
        // Arrays.fill(biases, 0);
        //Arrays.fill(runningMean, 0);
        Arrays.fill(runningVar, 1);

    }

    public static void addMatrix(double[][][] a, double[][] b) {

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                for (int k = 0; k < a[0][0].length; k++) {
                    a[i][j][k] += b[j][k];

                }
            }
        }


    }

    public static void addMatrix(double[][][] a, double[] b) {

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                for (int k = 0; k < a[0][0].length; k++) {

                    a[i][j][k] += b[k];

                }
            }
        }


    }

    public double[] sqrt_array(double[] a) {
        double[] out = new double[a.length];

        for (int i = 0; i < a.length; i++) {
            out[i] = Math.sqrt(a[i]);
        }
        return out;
    }

    public double[][][][] forward(double[][][][] inputs) {


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


        this.minus_mean = new double[inputs.length][inputs[0].length][inputs[0][0].length][inputs[0][0][0].length];


        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                for (int k = 0; k < inputs[0][0].length; k++) {
                    for (int l = 0; l < inputs[0][0][0].length; l++) {
                        minus_mean[i][j][k][l] = (inputs[i][j][k][l] - mean[j]);
                    }

                }
            }
        }

        standart_inputs = minus_mean.clone();
        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                for (int k = 0; k < inputs[0][0].length; k++) {
                    for (int l = 0; l < inputs[0][0][0].length; l++) {
                        standart_inputs[i][j][k][l] /= stddev[j];
                    }

                }

            }
        }

        double[][][][] out = Array_utils.copyArray(this.standart_inputs);

        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                for (int k = 0; k < inputs[0][0].length; k++) {
                    for (int l = 0; l < inputs[0][0][0].length; l++) {

                        out[i][j][k][l] = out[i][j][k][l] * gamma[j] + biases[j];

                    }
                }
            }
        }

        return out;
    }

    public double[][][][] backwardNew(double[][][][] grad_inputs) {
        int N = grad_inputs[0].length;

        double[] std = this.stddev;
        double[][][][] x_centered = minus_mean;
        double[][][][] x_norm = standart_inputs;

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
        return dx;
    }


    public double[][][][] backward(double[][][][] grad_inputs) {

        //int n,m,o,p = 0,0,0,0;

        int batch_size = grad_inputs.length;
        double[][][][] standard_grad = Array_utils.zerosLike(grad_inputs);

        double[] stddev_inv = Array_utils.div_matrix_by_scalarRE(this.stddev, 1);

        double[] var_pow = zerosLike(this.var);
        for (int i = 0; i < var.length; i++) {
            var_pow[i] = Math.pow(var[i], -3 / 2);
        }


        double[][][][] aux_x_minus_mean = Array_utils.zerosLike(minus_mean);
        double[][][][] var_grad_tmp = Array_utils.zerosLike(grad_inputs);
        double[][][][] mean_grad_part1 = Array_utils.zerosLike(standard_grad);


        for (int i = 0; i < grad_inputs.length; i++) {
            for (int j = 0; j < grad_inputs[0].length; j++) {
                for (int k = 0; k < grad_inputs[0][0].length; k++) {
                    for (int l = 0; l < grad_inputs[0][0][0].length; l++) {

                        standard_grad[i][j][k][l] = grad_inputs[i][j][k][l] * gamma[j];
                        var_grad_tmp[i][j][k][l] = standard_grad[i][j][k][l] * this.minus_mean[i][j][k][l] * -0.5 * var_pow[j];

                        aux_x_minus_mean[i][j][k][l] = 2 * minus_mean[i][j][k][l] / batch_size;

                        mean_grad_part1[i][j][k][l] = standard_grad[i][j][k][l] * (-stddev_inv[j]);

                    }

                }
            }
        }

        double[] var_grad = Array_utils.sum_axis_0_2_3(var_grad_tmp);


        double[] neg_sum_aux = Array_utils.neg_sum_axis_0_2_3(aux_x_minus_mean);
        double[] mean_grad_ = Array_utils.multiply1D(var_grad, neg_sum_aux);


        double[] mean_grad = Array_utils.sum_axis_0_2_3(mean_grad_part1);


        Array_utils.addMatrix(mean_grad, mean_grad_);


        double[][][][] gammaGradTmp = Utils.multiply(grad_inputs, standart_inputs);

        biasesGrad = Array_utils.sum_axis_0_2_3(grad_inputs);
        gammaGrad = Array_utils.sum_axis_0_2_3(gammaGradTmp);


        double[][][][] out = Array_utils.zerosLike(grad_inputs);
        for (int i = 0; i < standard_grad.length; i++) {
            for (int j = 0; j < standard_grad[0].length; j++) {
                for (int k = 0; k < out[0][0].length; k++) {
                    for (int l = 0; l < out[0][0][0].length; l++) {
                        out[i][j][k][l] = standard_grad[i][j][k][l] * stddev_inv[j] + var_grad[j] * aux_x_minus_mean[i][j][k][l] + mean_grad[j] / batch_size;
                    }

                }
            }

        }
        return out;


    }

    public double[][][][] backward(double[][][][] grad_inputs, double learningRate) {

        double[][][][] out = this.backwardNew(grad_inputs);
        this.updateParameter(learningRate);
        return out;

    }

    public void updateParameter(double learningRate) {
        // gamma -= gammaGrad * learningRate
        Utils.updateParameter(this.gamma, this.gammaGrad, learningRate);
        Utils.updateParameter(this.biases, this.biasesGrad, learningRate);


    }

}
