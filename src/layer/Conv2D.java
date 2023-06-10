package layer;

import utils.Array_utils;
import utils.Utils;

import static utils.Utils.genRandomWeight;

//https://github.com/detkov/Convolution-From-Scratch/blob/main/convolution.py
public class Conv2D {

    double[][][][] inputs;
    double[][][] input;
    int kernelSize2 = 3;

    int channels = 1;

    int kernelSize = 3;
    int num_filters;

    double[][][][] filters;

    double[] biases;

    public Conv2D(int num_filters) {
        this.num_filters = num_filters;
        this.filters = new double[kernelSize][kernelSize2][1][num_filters];
        this.biases = Utils.getOnesBiases(num_filters);
        genRandomWeight(this.filters);
        channels = 1;
    }

    public Conv2D(int num_filters, int channels) {
        this.num_filters = num_filters;
        this.filters = new double[kernelSize][kernelSize2][channels][num_filters];
        this.biases = Utils.getOnesBiases(num_filters);
        genRandomWeight(this.filters);
        this.channels = channels;
    }


    public Conv2D(int num_filters, int channels, int kernel_size) {
        this.num_filters = num_filters;
        this.kernelSize = kernel_size;
        this.kernelSize2 = kernel_size;
        this.filters = new double[kernelSize][kernelSize2][channels][num_filters];

        int[] shape = new int[]{kernel_size, kernel_size, channels, num_filters};
        genRandomWeight(this.filters);
        this.biases = Utils.getOnesBiases(num_filters);
        this.channels = channels;
    }


    public Conv2D(int num_filters, int channels, int kernelSize, int kernelSize2) {
        this.num_filters = num_filters;
        this.filters = new double[kernelSize][kernelSize2][channels][num_filters];
        this.kernelSize = kernelSize;
        this.kernelSize2 = kernelSize2;
        this.biases = Utils.getOnesBiases(num_filters);
        genRandomWeight(this.filters);
        this.channels = channels;

    }

    public static double[][][][] reshapeImg(double[][][][] a) {

        double[][][][] c = new double[a.length][a[0][0][0].length][a[0].length][a[0][0].length];
        for (int b = 0; b < a.length; b++) {
            for (int ci = 0; ci < a[0][0][0].length; ci++) {
                for (int i = 0; i < a[0].length; i++) {
                    for (int j = 0; j < a[0][0].length; j++) {
                        c[b][ci][i][j] = a[b][i][j][ci];
                    }
                }
            }


        }

        return c;
    }

    public static double[][][] reshapeImg(double[][][] a) {

        double[][][] c = new double[a[0][0].length][a.length][a[0].length];

        for (int ci = 0; ci < a[0][0].length; ci++) {
            for (int i = 0; i < a.length; i++) {
                for (int j = 0; j < a[0].length; j++) {
                    c[ci][i][j] = a[i][j][ci];
                }
            }


        }

        return c;
    }

    public static double[][][][] backReshapeImg(double[][][][] a) {

        double[][][][] c = new double[a.length][a[0][0][0].length][a[0][0].length][a[0].length];
        for (int b = 0; b < a.length; b++) {
            for (int ci = 0; ci < a[0].length; ci++) {
                for (int i = 0; i < a[0][0].length; i++) {
                    for (int j = 0; j < a[0][0][0].length; j++) {
                        c[b][i][j][ci] = a[b][ci][i][j];
                    }
                }
            }
        }
        return c;
    }


    public double sum_axis(double[][] imageRegion) {

        double out = 0;

        for (int ci = 0; ci < this.filters[0][0].length; ci++) {
            for (int f = 0; f < this.num_filters; f++) {
                for (int i = 0; i < this.kernelSize; i++) {
                    for (int j = 0; j < this.kernelSize2; j++) {
                        out += this.filters[i][j][ci][f];
                    }
                }
            }
        }
        return out;
    }

    public double[][][] getFilter(int ci) {

        double[][][] c = new double[kernelSize][kernelSize2][this.num_filters];

        for (int i = 0; i < kernelSize; i++) {
            for (int j = 0; j < kernelSize2; j++) {
                for (int k = 0; k < num_filters; k++) {
                    c[i][j][k] = this.filters[i][j][ci][k];
                }
            }
        }
        return c;
    }

    public double[] sum_axis_1_2(double[][] imageRegion, int ci_in) {

        double[] out = new double[this.num_filters];

        for (int f = 0; f < this.num_filters; f++) {
            for (int i = 0; i < this.kernelSize; i++) {
                for (int j = 0; j < this.kernelSize2; j++) {
                    out[f] += this.filters[i][j][ci_in][f];
                }
            }
            out[f] += this.biases[f];
        }
        return out;
    }


    public double[][][][] forward(double[][][][] inputs) {

        this.inputs = inputs;
        int h = inputs[0].length;
        int w = inputs[0][0].length;
        int C = inputs[0][0][0].length;


        int h_ = (h - this.kernelSize) + 1;
        int w_ = (w - this.kernelSize2) + 1;

        inputs = reshapeImg(inputs);

        double[][][][] outputs_2D = new double[inputs.length][h_][w_][num_filters];
        for (int b = 0; b < inputs.length; b++) {

            for (int ci = 0; ci < C; ci++) {
                double[][] imgRegion;
                for (int i = 0; i < h_; i++) {
                    for (int j = 0; j < w_; j++) {
                        imgRegion = Array_utils.getSubmatrix(inputs[b][ci], i, i + kernelSize, j, j + kernelSize2);
                        outputs_2D[b][i][j] = this.sum_axis_1_2(imgRegion, ci);
                    }
                }
            }
        }

        return outputs_2D;
    }

    public double[][][] forward(double[][][] input) {

        this.input = input;
        int h = input[0].length;
        int w = input[0][0].length;

        int h_ = (h - this.kernelSize) + 1;
        int w_ = (w - this.kernelSize2) + 1;

        input = reshapeImg(input);

        double[][][] outputs_2D = new double[h_][w_][num_filters];

        for (int ci = 0; ci < input[0][0].length; ci++) {
            double[][] imgRegion;
            for (int i = 0; i < h_; i++) {
                for (int j = 0; j < w_; j++) {
                    imgRegion = Array_utils.getSubmatrix(input[ci], i, i + kernelSize, j, j + kernelSize2);
                    outputs_2D[i][j] = this.sum_axis_1_2(imgRegion, ci);
                }
            }
        }

        return outputs_2D;
    }


    public double[][][][] backward(double[][][][] delta_out, double learningRate) {

        int h = this.inputs[0].length;
        int w = this.inputs[0][0].length;

        int h_ = (h - this.kernelSize) + 1;
        int w_ = (w - this.kernelSize2) + 1;

        double[][][][] delta_filters = new double[this.inputs.length][this.num_filters][this.kernelSize][this.kernelSize2];
        double[][] imgRegion;

        for (int b = 0; b < this.inputs.length; b++) {
            for (int i = 0; i < h_; i++) {
                for (int j = 0; j < w_; j++) {
                    for (int f = 0; f < this.num_filters; f++) {
                        imgRegion = Array_utils.getSubmatrix(inputs[b][f], i, i + kernelSize,
                                j, j + kernelSize2);
                        Utils.cal_matrix_mult_scalar(imgRegion, delta_out[b][i][j][f]);
                        Utils.addMatrix(delta_filters[b][f], imgRegion);

                    }

                }
            }
        }


        return delta_filters;
    }
}
