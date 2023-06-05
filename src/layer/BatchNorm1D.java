package layer;

import utils.Array_utils;
import utils.Utils;

import java.util.Arrays;

public class BatchNorm1D {

    double epsilon = 1e-8;
    double[] gamma;
    double[] gammaGrad;
    double[] biases;
    double[] biasesGrad;


    double[] stddev;
    double[] var;
    double[][] standart_inputs;
    double[][] minus_mean;

    public BatchNorm1D(int input_size) {
        this.gamma = new double[input_size];
        this.biases = new double[input_size];
        Arrays.fill(gamma, 1);
        Arrays.fill(biases, 1);

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
        //normalize varianz.
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

        standart_inputs = minus_mean.clone();
        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                standart_inputs[i][j] /= stddev[j];
            }
        }

        double[][] out = this.standart_inputs.clone();
        Utils.dotProdukt_1D(out, gamma);
        Utils.addMatrix(out, biases);
        return out;
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
        double[][] gammaGradTmp = Utils.matmul2D(grad_inputs, standart_inputs);


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
