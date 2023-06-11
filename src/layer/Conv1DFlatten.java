package layer;

import utils.Array_utils;
import utils.Utils;

public class Conv1DFlatten {

    int kernel_size = 3;
    double[][][] filters;
    int num_filters;
    boolean useBiases = false;
    double[][][] biases;
    double[][][] inputs_2D;
    double[][] input_2D;


    public Conv1DFlatten(int kernel_size, int filters, int[] input_shape) {
        this.kernel_size = kernel_size;
        this.filters = Utils.genRandomWeight(new int[]{filters, kernel_size, kernel_size});
        this.num_filters = filters;
        this.biases = new double[input_shape[0]][input_shape[1]][filters];

    }

    public Conv1DFlatten(int filters) {
        this.filters = Utils.genRandomWeight(new int[]{filters, kernel_size, kernel_size});
        this.num_filters = filters;
    }

    public double[] sum_axis_1_2(double[][] imageRegion) {

        double[] out = new double[this.num_filters];

        for (int f = 0; f < this.num_filters; f++) {
            for (int i = 0; i < this.kernel_size; i++) {
                for (int j = 0; j < this.kernel_size; j++) {
                    out[f] += this.filters[f][i][j];
                }
            }
        }
        return out;
    }


    public double[] forward(double[][] input) {
        /*
        automatically uses Flatten.
         */
        this.input_2D = input;
        int h = input.length;
        int w = input[0].length;

        int h_ = (h - this.kernel_size) + 1;
        int w_ = (w - this.kernel_size) + 1;

        double[][][] output_2D = new double[h_][w_][num_filters];
        if (useBiases) {
            output_2D = this.biases.clone();
        }

        double[][] imgRegion;
        for (int i = 0; i < h_; i++) {
            for (int j = 0; j < w_; j++) {
                imgRegion = Array_utils.getSubmatrix(input, i, i + kernel_size, j, j + kernel_size);
                output_2D[i][j] = this.sum_axis_1_2(imgRegion);


            }

        }
        return Array_utils.flatten(output_2D);
    }

    public double[][] forward(double[][][] inputs) {
        /*
        automatically uses Flatten.
         */
        this.inputs_2D = inputs;
        int h = inputs[0].length;
        int w = inputs[0][0].length;

        int h_ = (h - this.kernel_size) + 1;
        int w_ = (w - this.kernel_size) + 1;

        double[][][][] outputs_2D = new double[inputs.length][h_][w_][num_filters];
        for (int b = 0; b < inputs.length; b++) {
            double[][] imgRegion;
            for (int i = 0; i < h_; i++) {
                for (int j = 0; j < w_; j++) {
                    imgRegion = Array_utils.getSubmatrix(inputs[b], i, i + kernel_size, j, j + kernel_size);
                    outputs_2D[b][i][j] = this.sum_axis_1_2(imgRegion);
                }
            }
        }

        double[][] out = new double[outputs_2D.length][];
        for (int i = 0; i < outputs_2D.length; i++) {
            out[i] = Array_utils.flatten(outputs_2D[i]);
        }

        return out;
    }

    public double[][][] backward(double[] delta_outFlatten, double learningRate) {

        int h = this.input_2D.length;
        int w = this.input_2D[0].length;


        int h_ = (h - this.kernel_size) + 1;
        int w_ = (w - this.kernel_size) + 1;

        double[][][] delta_out = Utils.reshape(delta_outFlatten, new int[]{h_, w_, this.num_filters});

        double[][][] delta_filters = new double[this.num_filters][this.kernel_size][this.kernel_size];
        double[][] imgRegion;

        for (int i = 0; i < h_; i++) {
            for (int j = 0; j < w_; j++) {
                for (int f = 0; f < this.num_filters; f++) {
                    imgRegion = Array_utils.getSubmatrix(input_2D, i, i + kernel_size,
                            j, j + kernel_size);
                    Utils.cal_matrix_mult_scalar(imgRegion, delta_out[i][j][f]);
                    Utils.addMatrix(delta_filters[f], imgRegion);

                }

            }
        }
        Utils.updateParameter(this.filters, delta_filters, learningRate);
        return delta_filters;
    }


}
