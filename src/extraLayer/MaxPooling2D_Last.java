package extraLayer;

import utils.Array_utils;

import java.util.Arrays;

public class MaxPooling2D_Last {

    int stride1 = 2;
    int stride2 = 2;
    int kernelSize1 = 2;
    int kernelSize2 = 2;

    int[][][] inputMaxIndicies_R;
    int[][][] inputMaxIndicies_C;

    int inputHeight = 26;
    int inputWidth = 26;

    int channels;

    int outputHeight;
    int outputWidth;
    double[][][] backward;
    double[][][][] backwards;

    public MaxPooling2D_Last() {

        outputHeight = (((inputHeight - kernelSize1) / (stride1)) + 1);
        outputWidth = (((inputWidth - kernelSize2) / (stride2)) + 1);


    }

    public MaxPooling2D_Last(int strides) {

        this.stride1 = strides;
        this.stride2 = strides;

        outputHeight = (((inputHeight - kernelSize1) / (stride1)) + 1);
        outputWidth = (((inputWidth - kernelSize2) / (stride2)) + 1);


    }

    public MaxPooling2D_Last(int[] strides, int[] kernelSizes) {

        this.stride1 = strides[0];
        this.stride2 = strides[0];

        this.kernelSize1 = kernelSizes[0];
        this.kernelSize2 = kernelSizes[1];

        outputHeight = (((inputHeight - kernelSize1) / (stride1)) + 1);
        outputWidth = (((inputWidth - kernelSize2) / (stride2)) + 1);


    }

    public MaxPooling2D_Last(int[] shape) {


        inputHeight = shape[0];
        inputWidth = shape[1];
        channels = shape[2];

        outputHeight = (((inputHeight - kernelSize1) / (stride1)) + 1);
        outputWidth = (((inputWidth - kernelSize2) / (stride2)) + 1);


    }

    public void printOutputShape() {
        System.out.println(Arrays.toString(getOutputShape()));

    }

    public int[] getOutputShape() {

        return new int[]{outputHeight, outputWidth, channels};
    }


    public double[] getMultiSum(int pos, double[][][] in, int h_st, int h_end, int w_st, int w_end) {

        double[] out = new double[3];


        for (int i = 0; i < h_end - h_st; i++) {
            for (int j = 0; j < w_end - w_st; j++) {
                if (out[0] < in[h_st + i][w_st + j][pos]) {
                    out[0] = in[h_st + i][w_st + j][pos];
                    out[1] = h_st + i;
                    out[2] = w_st + j;

                }

            }
        }


        return out;
    }

    public double[][][][] forward(double[][][][] inputs) {


        int channels = inputs[0][0][0].length;
        double[][][][] output = new double[inputs.length][outputHeight][outputWidth][channels];

        backwards = Array_utils.copyArray(inputs);

        double[] tmp;
        for (int bs = 0; bs < inputs.length; bs++) {

            for (int i = 0; i < outputHeight; i++) {
                for (int j = 0; j < outputWidth; j++) {
                    for (int ci = 0; ci < channels; ci++) {

                        tmp = getMultiSum(ci, inputs[bs], i * (stride1), kernelSize1 + (i * (stride1)),
                                j * (stride2), kernelSize2 + (j * (stride2)));

                        output[bs][i][j][ci] = tmp[0];

                        int h = (int) tmp[1];
                        int w = (int) tmp[2];


                        backwards[bs][h][w][ci] = tmp[0];


                    }
                }

            }
        }

        return output;
    }

    public double[][][] forward(double[][][] input) {

        int channels = input[0][0].length;
        double[][][] output = new double[outputHeight][outputWidth][channels];

        //backward = Array_utils.copyArray(input);
        inputMaxIndicies_R = new int[outputHeight][outputWidth][channels];
        inputMaxIndicies_C = new int[outputHeight][outputWidth][channels];

        double[] tmp;

        for (int i = 0; i < outputHeight; i++) {
            for (int j = 0; j < outputWidth; j++) {
                for (int ci = 0; ci < channels; ci++) {

                    tmp = getMultiSum(ci, input, i * (stride1), kernelSize1 + (i * (stride1)),
                            j * (stride2), kernelSize2 + (j * (stride2)));

                    output[i][j][ci] = tmp[0];

                    int h = (int) tmp[1];
                    int w = (int) tmp[2];
                    inputMaxIndicies_R[i][j][ci] = h;
                    inputMaxIndicies_C[i][j][ci] = w;


                }

            }
        }
        return output;
    }

    public double[][][] backward(double[][][] grad) {

        /**
         * channel position last Dim.
         */
        double[][][] grad_output = new double[inputHeight][inputWidth][channels];


        for (int i = 0; i < grad[0][0].length; i++) {
            for (int r = 0; r < outputHeight; r++) {
                for (int c = 0; c < outputWidth; c++) {
                    int max_i = inputMaxIndicies_R[r][c][i];
                    int max_j = inputMaxIndicies_C[r][c][i];

                    if (max_i != -1) {
                        grad_output[max_i][max_j][i] += grad[r][c][i];
                    }

                }
            }
        }

        return grad_output;

    }

    public double[][][][] backward(double[][][][] c) {
        return this.backwards;

    }

    public double[][][] backward(double[][][] c, double learningRate) {
        return this.backward;

    }

    public double[][][][] backward(double[][][][] c, double learningRate) {
        return this.backwards;

    }


}
