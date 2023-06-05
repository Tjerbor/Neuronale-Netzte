package layer;

import utils.Array_utils;
import utils.Utils;

import java.util.Arrays;

import static utils.Array_utils.zerosLike;

public class BatchNorm2D {

    double epsilon = 1e-8;
    double[][] gamma;
    double[][] gammaGrad;
    double[][] biases;
    double[][] biasesGrad;

    double[][] stddev;
    double[][] var;
    double[][][] standart_inputs;
    double[][][] minus_mean;

    public BatchNorm2D(int inputSize, int inputSize2) {
        this.gamma = new double[inputSize][inputSize2];
        this.biases = new double[inputSize][inputSize2];
        Arrays.fill(gamma, 1);
        Arrays.fill(biases, 0);

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


        double[][] mean = Array_utils.mean_axis_0(inputs);
        this.var = Array_utils.var_axis_0(inputs);
        //normalize varianz.
        for (int i = 0; i < var.length; i++) {
            for (int j = 0; j < var[0].length; j++) {
                var[i][j] += epsilon;
            }

        }

        this.stddev = sqrt_array(var);


        this.minus_mean = new double[inputs.length][inputs[0].length]
                [inputs[0][0].length];


        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                for (int k = 0; k < inputs[0][0].length; k++) {
                    minus_mean[i][j][k] = (inputs[i][j][k] - mean[j][k]);
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
        Array_utils.addMatrix(out, biases);
        return out;
    }

    public double[][][] backward(double[][][] grad_inputs) {

        int batch_size = grad_inputs.length;
        double[][][] standard_grad = Utils.matmul3D_2D(grad_inputs, gamma);

        double[][] stddev_inv = Array_utils.div_matrix_by_scalarRE(this.stddev, 1);

        double[][] var_pow = zerosLike(this.var);
        for (int i = 0; i < var.length; i++) {
            for (int j = 0; j < var[0].length; j++) {
                var_pow[i][j] = Math.pow(var[i][j], -3 / 2);
            }

        }


        double[][][] var_grad_tmp = Array_utils.zerosLike(grad_inputs);
        for (int i = 0; i < grad_inputs.length; i++) {
            for (int j = 0; j < grad_inputs[0].length; j++) {
                for (int k = 0; k < grad_inputs[0][0].length; k++) {
                    var_grad_tmp[i][j][k] = standard_grad[i][j][k] * this.minus_mean[i][j][k] * -0.5 * var_pow[j][k];
                }
            }
        }

        double[][] var_grad = Array_utils.sum_axis_0(var_grad_tmp);

        double[][][] aux_x_minus_mean = Array_utils.zerosLike(minus_mean);
        for (int i = 0; i < minus_mean.length; i++) {
            for (int j = 0; j < minus_mean[0].length; j++) {
                for (int k = 0; k < minus_mean[0][0].length; k++) {
                    aux_x_minus_mean[i][j][k] = 2 * minus_mean[i][j][k] / batch_size;
                }

            }
        }

        double[][] neg_sum_aux = Array_utils.neg_sum_axis_0(aux_x_minus_mean);
        double[][] mean_grad_ = Array_utils.multiply2D(var_grad, neg_sum_aux);

        double[][][] mean_grad_part1 = Array_utils.zerosLike(standard_grad);
        for (int i = 0; i < standard_grad.length; i++) {
            for (int j = 0; j < standard_grad[0].length; j++) {
                for (int k = 0; k < standard_grad[0][0].length; k++) {
                    mean_grad_part1[i][j][k] = standard_grad[i][j][k] * (-stddev_inv[j][k]);
                }
            }
        }

        double[][] mean_grad = Array_utils.sum_axis_0(mean_grad_part1);


        Array_utils.addMatrix(mean_grad, mean_grad_);


        double[][][] gammaGradTmp = Utils.matmul3D(grad_inputs, standart_inputs);


        biasesGrad = Array_utils.sum_axis_0(grad_inputs);
        gammaGrad = Array_utils.sum_axis_0(gammaGradTmp);


        double[][][] out = Array_utils.zerosLike(grad_inputs);
        for (int i = 0; i < standard_grad.length; i++) {
            for (int j = 0; j < standard_grad[0].length; j++) {
                for (int k = 0; k < out[0][0].length; k++) {
                    out[i][j][k] = standard_grad[i][j][k] * stddev_inv[j][k] + var_grad[j][k]
                            * aux_x_minus_mean[i][j][k] + mean_grad[j][k] / batch_size;
                }
            }

        }


        return out;


    }

    public double[][][] backward(double[][][] grad_inputs, double learningRate) {

        double[][][] out = this.backward(grad_inputs);
        this.updateParameter(learningRate);
        return out;

    }

    public void updateParameter(double learningRate) {
        // gamma -= gammaGrad * learningRate
        Utils.updateParameter(this.gamma, this.gammaGrad, learningRate);
        Utils.updateParameter(this.biases, this.biasesGrad, learningRate);


    }

}
