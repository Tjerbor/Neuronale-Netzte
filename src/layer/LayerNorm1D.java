package layer;

import utils.Array_utils;
import utils.Utils;

import java.util.Arrays;

public class LayerNorm1D {

    double epsilon = 1e-8;
    double[] gamma;
    double[] biases;


    double[] stddev;
    double[] var;
    double[][] standart_inputs;
    double[][] minus_mean;

    public LayerNorm1D(int input_size) {
        this.gamma = new double[input_size];
        this.biases = new double[input_size];
        Arrays.fill(gamma, 1);
        Arrays.fill(biases, 1);

    }

    public double[][] forward(double[][] inputs) {


        double[] mean = Utils.mean_axis_1(inputs);
        this.var = Array_utils.var_axis_1(inputs);
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
}
