package extraLayer;

import utils.ArrayMathUtils;
import utils.RandomUtils;

import java.util.Random;


public class Conv2D {

    final private int kernelSize1;
    final private int kernelSize2;
    final private int stride1;
    final private int stride2;
    final private int channels;
    final private int inputHeight;
    final private int inputWidth;
    final private int outputHeight;
    final private int outputWidth;
    final private int numFilters;
    final private int paddingH = 0;
    final private int paddingW = 0;

    private double learningRate = 1e-4;
    private double[][][][] weights;
    private double[][][] input;
    private double[][][][] inputs;

    public Conv2D(int numFilters, int kernelSize, int strideSize, int channels, int inputHeight, int inputWidth) {
        this.kernelSize1 = kernelSize;
        this.kernelSize2 = kernelSize;
        this.stride1 = strideSize;
        this.stride2 = strideSize;
        this.channels = channels;
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.numFilters = numFilters;

        outputHeight = (((inputHeight - kernelSize1 + (2 * paddingH)) / (stride1)) + 1);
        outputWidth = (((inputWidth - kernelSize2 + (2 * paddingW)) / (stride2)) + 1);

        weights = new double[kernelSize1][kernelSize2][channels][numFilters];
        generateRandomFilters(numFilters);

    }

    public Conv2D(int[] config) {
        if (config.length < 7) {
            throw new IllegalArgumentException();
        }

        this.numFilters = config[0];
        this.kernelSize1 = config[1];
        this.kernelSize2 = config[2];
        this.stride1 = config[3];
        this.stride2 = config[4];


        inputHeight = config[config.length - 3];
        inputWidth = config[config.length - 2];
        channels = config[config.length - 1];

        outputHeight = (((inputHeight - kernelSize1 + (2 * paddingH)) / (stride1)) + 1);
        outputWidth = (((inputWidth - kernelSize2 + (2 * paddingW)) / (stride2)) + 1);


        this.weights = new double[kernelSize1][kernelSize2][channels][numFilters];
        RandomUtils.genGaussianRandomWeight(weights);

    }

    public static double[][] flipud_fliplr(double[][] a) {

        int s1 = a.length;
        int s2 = a[0].length;

        double[][] c = new double[s1][s2];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                c[i][j] = a[s1 - 1 - i][s1 - 1 - j];
            }
        }
        return c;
    }

    public int[] getInputShape() {
        return new int[]{channels, inputHeight, inputWidth};
    }

    public int[] getOutputShape() {
        return new int[]{numFilters, outputHeight, outputWidth};
    }

    private void generateRandomFilters(int numFilters) {
        Random random = new Random();

        for (int l = 0; l < numFilters; l++) {
            for (int k = 0; k < 1; k++) {
                for (int i = 0; i < kernelSize1; i++) {
                    for (int j = 0; j < kernelSize1; j++) {


                        weights[i][j][k][l] = random.nextGaussian();
                    }
                }
            }
        }

    }

    public double[][][][] getWeights() {
        return weights;
    }

    public double[][][] forward(double[][][] input) {
        this.input = input;

        double[][][] output = new double[numFilters][][];

        for (int m = 0; m < input.length; m++) {
            for (int fi = 0; fi < this.numFilters; fi++) {
                output[fi] = convolveForward(input[m], weights, stride1, m, fi);
            }
        }

        return output;

    }

    public double[][][][] forward(double[][][][] inputs) {
        this.inputs = inputs;

        double[][][][] output = new double[inputs.length][numFilters][][];

        for (int bi = 0; bi < inputs.length; bi++) {
            for (int m = 0; m < inputs[0].length; m++) {
                for (int fi = 0; fi < this.numFilters; fi++) {
                    output[bi][fi] = convolveForward(inputs[bi][m], weights, stride1, m, fi);
                }


            }
        }

        return output;

    }

    private double[][] convolve(double[][] input, double[][] filter) {

        int outRows = (input.length - filter.length) / this.stride1 + 1;
        int outCols = (input[0].length - filter[0].length) / this.stride1 + 1;

        int inRows = input.length;
        int inCols = input[0].length;

        int fRows = filter.length;
        int fCols = filter[0].length;

        double[][] output = new double[outRows][outCols];

        int outRow = 0;
        int outCol;

        for (int i = 0; i <= inRows - fRows; i += stride1) {

            outCol = 0;

            for (int j = 0; j <= inCols - fCols; j += stride2) {

                double sum = 0.0;

                //Apply Filter around this position
                for (int x = 0; x < fRows; x++) {
                    for (int y = 0; y < fCols; y++) {
                        int inputHeight = i + x;
                        int inputWidth = j + y;

                        double value = filter[x][y] * input[inputHeight][inputWidth];
                        sum += value;
                    }
                }

                output[outRow][outCol] = sum;
                outCol++;
            }

            outRow++;

        }

        return output;

    }

    private double[][] convolveForward(double[][] input, double[][][][] filter, int stepSize, int m, int fi) {

        int outRows = (input.length - filter.length) / stepSize + 1;
        int outCols = (input[0].length - filter[0].length) / stepSize + 1;

        int inRows = input.length;
        int inCols = input[0].length;

        int fRows = filter.length;
        int fCols = filter[0].length;

        double[][] output = new double[outRows][outCols];

        int outRow = 0;
        int outCol;

        for (int i = 0; i <= inRows - fRows; i += stepSize) {

            outCol = 0;

            for (int j = 0; j <= inCols - fCols; j += stepSize) {

                double sum = 0.0;

                //Apply Filter around this position
                for (int x = 0; x < fRows; x++) {
                    for (int y = 0; y < fCols; y++) {
                        int inputRowIndex = i + x;
                        int inpurColIndex = j + y;

                        double value = filter[x][y][m][fi] * input[inputRowIndex][inpurColIndex];
                        sum += value;
                    }
                }

                output[outRow][outCol] = sum;
                outCol++;
            }

            outRow++;

        }

        return output;

    }

    public double[][] spaceArray(double[][] input) {

        if (stride1 == 1) {
            return input;
        }

        int outRows = (input.length - 1) * stride1 + 1;
        int outCols = (input[0].length - 1) * stride1 + 1;

        double[][] output = new double[outRows][outCols];

        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                output[i * stride1][j * stride1] = input[i][j];
            }
        }

        return output;
    }

    private double[][] fullConvolve(double[][] input, double[][] filter) {

        int outRows = (input.length + filter.length) + 1;
        int outCols = (input[0].length + filter[0].length) + 1;

        int inRows = input.length;
        int inCols = input[0].length;

        int fRows = filter.length;
        int fCols = filter[0].length;

        double[][] output = new double[outRows][outCols];

        int outRow = 0;
        int outCol;

        for (int i = -fRows + 1; i < inRows; i++) {

            outCol = 0;

            for (int j = -fCols + 1; j < inCols; j++) {

                double sum = 0.0;

                //Apply Filter around this position
                for (int x = 0; x < fRows; x++) {
                    for (int y = 0; y < fCols; y++) {
                        int inputRowIndex = i + x;
                        int inputColIndex = j + y;

                        if (inputRowIndex >= 0 && inputColIndex >= 0 && inputRowIndex < inRows && inputColIndex < inCols) {
                            double value = filter[x][y] * input[inputRowIndex][inputColIndex];
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


    public double[][] addSubmatrix(double[][][][] a, double[][] b, int channels, int f) {

        double[][] c = new double[b.length][b[0].length];

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                c[i][j] = a[i][j][channels][f] + b[i][j];
            }
        }

        return c;
    }


    public double[][][] backward(double[][][] gradInput) {


        double[][][][] filtersDelta = new double[kernelSize1][kernelSize1][channels][numFilters];

        double[][][] dLdOPreviousLayer = new double[channels][inputHeight][inputWidth];


        for (int i = 0; i < input.length; i++) {

            double[][] errorForInput = new double[inputHeight][inputWidth];

            for (int f = 0; f < numFilters; f++) {

                double[][] currFilter = new double[kernelSize1][kernelSize1];

                double[][] error = gradInput[i * weights.length + f];

                double[][] spacedError = spaceArray(error);
                double[][] dLdF = convolve(input[i], spacedError);

                double[][] delta = ArrayMathUtils.multiply(dLdF, learningRate * -1);
                double[][] newTotalDelta = addSubmatrix(filtersDelta, delta, i, f);


                for (int j = 0; j < kernelSize1; j++) {
                    for (int k = 0; k < kernelSize1; k++) {
                        filtersDelta[j][k][i][f] = newTotalDelta[j][k];
                        currFilter[j][k] = this.weights[j][k][i][f];
                    }
                }

                double[][] flippedError = flipud_fliplr(spacedError);
                errorForInput = ArrayMathUtils.add(errorForInput, fullConvolve(currFilter, flippedError));
            }

            dLdOPreviousLayer[i] = (errorForInput);

        }

        for (int i = 0; i < kernelSize1; i++) {
            for (int j = 0; j < kernelSize1; j++) {
                for (int k = 0; k < 1; k++) {
                    for (int l = 0; l < numFilters; l++) {
                        weights[i][j][k][l] += filtersDelta[i][j][k][l];
                    }
                }
            }
        }

        return dLdOPreviousLayer;

    }

    public double[][][] backward(double[][][] gradInput, double learningRate) {


        double[][][][] filtersDelta = new double[kernelSize1][kernelSize1][channels][numFilters];

        double[][][] dLdOPreviousLayer = new double[channels][inputHeight][inputWidth];


        for (int i = 0; i < input.length; i++) {

            double[][] errorForInput = new double[inputHeight][inputWidth];

            for (int f = 0; f < numFilters; f++) {

                double[][] currFilter = new double[kernelSize1][kernelSize1];

                double[][] error = gradInput[i * weights.length + f];

                double[][] spacedError = spaceArray(error);
                double[][] dLdF = convolve(input[i], spacedError);

                double[][] delta = ArrayMathUtils.multiply(dLdF, learningRate * -1);
                double[][] newTotalDelta = addSubmatrix(filtersDelta, delta, i, f);


                for (int j = 0; j < kernelSize1; j++) {
                    for (int k = 0; k < kernelSize1; k++) {
                        filtersDelta[j][k][i][f] = newTotalDelta[j][k];
                        currFilter[j][k] = this.weights[j][k][i][f];
                    }
                }

                double[][] flippedError = flipud_fliplr(spacedError);
                errorForInput = ArrayMathUtils.add(errorForInput, fullConvolve(currFilter, flippedError));
            }

            dLdOPreviousLayer[i] = (errorForInput);

        }

        for (int i = 0; i < kernelSize1; i++) {
            for (int j = 0; j < kernelSize1; j++) {
                for (int k = 0; k < 1; k++) {
                    for (int l = 0; l < numFilters; l++) {
                        weights[i][j][k][l] += filtersDelta[i][j][k][l];
                    }
                }
            }
        }

        return dLdOPreviousLayer;

    }

}
