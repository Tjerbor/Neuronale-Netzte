package layer;

import utils.Array_utils;
import utils.Utils;

/**
 * normalizes the Data from One Layer. this is why the data
 * can have a singe sample, because that sample  data is normalized.
 * is used in modern Architektur like Transformer (Multi-head-Attention).
 * Ein Beispiel kommt noch.
 */
public class LayerNorm1D {

    double epsilon = 1e-8;

    double momentum = 0.9;
    double[] gamma;
    double[] gammaGrad;
    double[] biases;
    double[] biasesGrad;


    double[] stddev;
    double[] var;
    double[][] standart_inputs;
    double[][] minus_mean;

    boolean training = false;
    boolean useMomentum = false;

    double[] runningMean;
    double[] runningVar;

    public LayerNorm1D(int input_size) {
        this.gamma = new double[input_size];
        this.runningMean = new double[input_size];
        this.runningVar = new double[input_size];
        Utils.fill(this.gamma, 1);
        Utils.fill(this.biases, 1);


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


    public double[][] forward(double[][] inputs) {


        double[] mean = Utils.mean_axis_1(inputs);
        this.var = Array_utils.var_axis_1(inputs);


        if (useMomentum) {
            for (int i = 0; i < mean.length; i++) {
                runningMean[i] = momentum * runningMean[i] + (1 - momentum) * mean[i];
                runningVar[i] = momentum * runningVar[i] + (1 - momentum) * var[i];
            }


        }

        //normalize varianz.
        for (int i = 0; i < var.length; i++) {
            var[i] += epsilon;
        }

        this.stddev = Array_utils.sqrtArrayRE(var);


        this.minus_mean = new double[inputs.length][inputs[0].length];


        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                minus_mean[i][j] = (inputs[i][j] - mean[j]);
            }
        }


        standart_inputs = Array_utils.copyArray(minus_mean);
        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                standart_inputs[i][j] /= stddev[j];
            }
        }

        double[][] out = Array_utils.copyArray(this.standart_inputs);
        Utils.multiply(out, gamma);
        Utils.addMatrix(out, biases);
        return out;
    }

    public double[][] backward(double[][] grad_inputs) {

        int N = grad_inputs.length;
        double[] std = this.stddev;
        double[][] x_centered = minus_mean;
        double[][] x_norm = standart_inputs;

        gammaGrad = Array_utils.sum_axis_1(Utils.multiply(grad_inputs, x_norm));
        biasesGrad = Array_utils.sum_axis_1(grad_inputs);


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


        double[] dstd = Array_utils.sum_axis_1(dstdTmp);
        //double[] dvar = dstd / 2 / std;
        double[] dvar = Array_utils.zerosLike(dstd);
        for (int i = 0; i < dstd.length; i++) {
            dvar[i] = dstd[i] / 2 / std[i];
        }

        double[][] dx = Array_utils.zerosLike(dx_centered);

        for (int i = 0; i < dx.length; i++) {
            for (int j = 0; j < dx[0].length; j++) {
                dx[i][j] = dx_centered[i][j] + (dmean[i] +
                        dvar[i] * 2 * x_centered[i][j]
                ) / N;
            }
        }


        return dx;


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

    public double[] forward(double[] input) {


        double mean = Utils.mean(input);
        this.var[0] = Array_utils.var(input);


        if (useMomentum) {
            for (int i = 0; i < runningMean.length; i++) {
                runningMean[i] = momentum * runningMean[i] + (1 - momentum) * mean;
                runningVar[i] = momentum * runningVar[i] + (1 - momentum) * var[i];
            }


        }

        var[0] += epsilon;

        this.stddev[0] = Math.sqrt(var[0]);

        this.minus_mean[0] = new double[input.length];


        for (int i = 0; i < input.length; i++) {
            minus_mean[0][i] = (input[i] - mean);
        }


        standart_inputs[0] = Array_utils.copyArray(minus_mean[0]);
        for (int i = 0; i < input.length; i++) {
            standart_inputs[0][i] /= stddev[i];
        }

        double[] out = Array_utils.copyArray(this.standart_inputs[0]);

        for (int i = 0; i < out.length; i++) {
            out[i] = out[i] * gamma[i] + biases[i];
        }
        return out;
    }


    public double[] backward(double[] grad_inputs) {

        int N = grad_inputs.length;
        double std = this.stddev[0];
        double[] x_centered = minus_mean[0];
        double[] x_norm = standart_inputs[0];

        gammaGrad = Utils.multiply(grad_inputs, x_norm);
        biasesGrad = grad_inputs;


        double[] dx_norm = Utils.multiply(grad_inputs, gamma);
        double[] dx_centered = Array_utils.div_matrix_by_scalarRE(dx_norm, std);

        double[] dmean = Array_utils.addMatrixScalar(dx_centered, 2 / N);
        dmean = Array_utils.mult_matrix_by_scalar(dmean, Array_utils.sum(x_centered));

        double[] dstdTmp = Array_utils.zerosLike(dx_norm);

        //(dx_norm * x_centered * -std**(-2))
        for (int i = 0; i < dx_norm.length; i++) {
            dstdTmp[i] = dx_norm[i] * x_centered[i] * -Math.pow(std, -2);
        }


        double dstd = Array_utils.sum(dstdTmp);
        //double[] dvar = dstd / 2 / std;
        double dvar = dvar = dstd / 2 / std;


        double[] dx = Array_utils.zerosLike(dx_centered);

        for (int i = 0; i < dx.length; i++) {

            dx[i] = dx_centered[i] + (dmean[i] +
                    dvar * 2 * x_centered[i]
            ) / N;
        }


        return dx;


    }


}
