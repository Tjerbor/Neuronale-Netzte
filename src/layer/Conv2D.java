package layer;

import utils.Array_utils;
import utils.Utils;

import java.util.Arrays;

//https://www.youtube.com/watch?v=Lakz2MoHy6o
//https://github.com/TheIndependentCode/Neural-Network/blob/master/convolutional.py


/**
 * The Depth of the Dimension is one the frist position. Like 1 for Grayscale Images.
 * (1 , 28, 28)
 */
public class Conv2D {


    int h;
    int depth;
    int w;
    int h_out;
    int w_out;

    boolean useBiases = false;
    double[][][][] kernels;
    double[][][] biases;
    double[][][] input;

    int kernelSize = 3;

    int kernelSize2 = 3;
    int num_filters = 1;


    public Conv2D(int num_filters, int[] shape, int kernelSize) {
        this.num_filters = num_filters;


        this.kernelSize = kernelSize;
        this.kernelSize2 = kernelSize;
        this.depth = shape[0];
        this.h = shape[1];
        this.w = shape[2];

        kernels = new double[num_filters][shape[0]][kernelSize][kernelSize2];
        Utils.genRandomWeight(kernels);

        this.h_out = h - kernelSize + 1;
        this.w_out = w - kernelSize2 + 1;

        if (useBiases) {
            biases = new double[num_filters][h_out][w_out];
            Utils.genRandomWeight(biases);
        }
    }

    public Conv2D(int num_filters, int[] shape) {
        this.num_filters = num_filters;
        kernels = new double[num_filters][shape[0]][kernelSize][kernelSize2];
        Utils.genRandomWeight(kernels);

        this.depth = shape[0];
        this.h = shape[1];
        this.w = shape[2];

        this.h_out = h - kernelSize + 1;
        this.w_out = w - kernelSize2 + 1;


    }

    public static void main(String[] args) {


        Conv2D C = new Conv2D(8, new int[]{1, 6, 6}, 3);

        double[][] a = new double[][]{{0, 1, 2,},
                {3, 4, 5},
                {6, 7, 8}};

        double[][] b = new double[6][6];

        int count = 0;
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                b[i][j] = count;
                count += 1;
            }
        }

        System.out.println(Arrays.deepToString(b));
        double[][] c = C.correlate2D(b, a);
        System.out.println(Arrays.deepToString(c));

        System.out.println(Arrays.deepToString(convolve2DFull(c, a)));
    }

    public static double[][] convolve2DFull(double[][] grad_input, double[][] kernel) {

        kernel = Array_utils.flipud_fliplr(kernel);
        int outRows = (grad_input.length + kernel.length) - 1;
        int outCols = (grad_input[0].length + kernel[0].length) - 1;

        int gradH = grad_input.length;
        int gradW = grad_input[0].length;

        int kerN = kernel.length;
        int kerM = kernel[0].length;

        double[][] output = new double[outRows][outCols];

        int outRow = 0;
        int outCol;

        for (int i = -kerN + 1; i < gradH; i++) {

            outCol = 0;

            for (int j = -kerM + 1; j < gradW; j++) {

                double sum = 0.0;

                //Apply Filter around this position
                for (int x = 0; x < kerN; x++) {
                    for (int y = 0; y < kerM; y++) {
                        int inputRowIndex = i + x;
                        int inputColIndex = j + y;

                        if (inputRowIndex >= 0 && inputColIndex >= 0 && inputRowIndex < gradH && inputColIndex < gradW) {
                            double value = kernel[x][y] * grad_input[inputRowIndex][inputColIndex];
                            sum += value;
                        }
                    }
                }

                output[outRow][outCol] = sum;
                outCol++;
            }

            outRow++;

        }

        return output;
    }

    private void genBiases() {
        biases = new double[num_filters][h_out][w_out];
        Utils.genRandomWeight(biases);
    }

    public void setUseBiases(boolean useBiases) {
        this.useBiases = useBiases;
        if (useBiases) {
            genBiases();
        } else {
            this.biases = null;
        }

    }

    /**
     * Does not uses biases
     *
     * @param grad_input
     * @param learningRate
     */
    public double[][][] backward(double[][][] grad_input, double learningRate) {

        int depth = this.input.length;

        double[][][] grad_out = Array_utils.zerosLike(this.input);
        double[][] tmp;

        double[][][][] dkernels = new double[num_filters][depth][kernelSize][kernelSize2];
        for (int i = 0; i < num_filters; i++) {
            for (int j = 0; j < depth; j++) {
                dkernels[i][j] = correlate2D(this.input[j], grad_input[j]);
                tmp = convolve2DFull(this.input[j], kernels[i][j]);
                Array_utils.addMatrix(grad_out[j], tmp);

            }
        }


        Utils.updateParameter(kernels, dkernels, learningRate);

        if (useBiases) {
            Utils.updateParameter(biases, grad_out, learningRate);
        }
        return grad_out;

    }

    public double[][] correlate2D(double[][] img, double[][] kernel) {

        int h = img.length;
        int w = img[0].length;


        int kernelSize = kernel.length;
        int kernelSize2 = kernel[0].length;

        int h_ = (h - kernelSize) + 1;
        int w_ = (w - kernelSize2) + 1;


        double sum;
        double[][] output = new double[h_][w_];
        for (int i = 0; i < h_; i++) {
            for (int j = 0; j < w_; j++) {
                sum = 0;
                for (int ki = 0; ki < kernelSize; ki++) {
                    for (int kj = 0; kj < kernelSize2; kj++) {

                        sum += img[ki + i][j + kj] * kernel[ki][kj];

                    }
                }
                output[i][j] = sum;
            }
        }


        return output;
    }

    public double[][][] forward(double[][][] img) {

        this.input = img;
        int h = img[0].length;
        int w = img[0][0].length;

        int h_ = (h - this.kernelSize) + 1;
        int w_ = (w - this.kernelSize2) + 1;

        int depth = img.length;
        double[][][] output;

        if (useBiases) {
            output = Array_utils.copyArray(this.biases);
        } else {
            output = new double[num_filters][h_][w_];
        }

        for (int i = 0; i < num_filters; i++) {
            for (int j = 0; j < depth; j++) {
                output[i] = correlate2D(img[j], kernels[i][j]);

            }
        }
        return output;
    }


}
