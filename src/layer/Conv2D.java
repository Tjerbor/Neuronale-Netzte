package layer;

import function.Activation;
import function.Dropout;
import optimizer.Optimizer;
import utils.ArrayMathUtils;
import utils.Matrix;
import utils.RandomUtils;

import java.util.Arrays;
import java.util.Random;

import static load.writeUtils.writeWeights;
import static utils.Array_utils.reFlat;


/**
 * Convolution expects the input Dim to be (channels, Heights, Width)
 * Is important because the Output Shape is (numFilter, OutputHeight, OutputWidth)
 */

public class Conv2D extends Layer {

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
    boolean training = true;
    boolean useBiases = false;

    Optimizer optimizer;

    private double learningRate;
    private double[][][][] weights;
    private double[][][] biases;
    private double[][][] input;
    private double[][][][] inputs;


    public Conv2D(int[] shape, int numFilters, int kernelSize, int strideSize) {
        this.kernelSize1 = kernelSize;
        this.kernelSize2 = kernelSize;
        this.stride1 = strideSize;
        this.stride2 = strideSize;
        this.channels = shape[0];
        this.inputHeight = shape[1];
        this.inputWidth = shape[2];


        outputHeight = (((inputHeight - kernelSize1 + (2 * paddingH)) / (stride1)) + 1);
        outputWidth = (((inputWidth - kernelSize2 + (2 * paddingW)) / (stride2)) + 1);

        this.numFilters = numFilters;
        weights = new double[kernelSize1][kernelSize2][channels][numFilters];
        RandomUtils.genTypeWeights(2, weights);

    }

    public Conv2D(int[] shape, int numFilters, int[] kernelSize, int[] strideSize) {
        this.kernelSize1 = kernelSize[0];
        this.kernelSize2 = kernelSize[1];
        this.stride1 = strideSize[0];
        this.stride2 = strideSize[1];
        this.channels = shape[0];
        this.inputHeight = shape[1];
        this.inputWidth = shape[2];
        this.numFilters = numFilters;

        outputHeight = (((inputHeight - kernelSize1 + (2 * paddingH)) / (stride1)) + 1);
        outputWidth = (((inputWidth - kernelSize2 + (2 * paddingW)) / (stride2)) + 1);

        weights = new double[kernelSize1][kernelSize2][channels][numFilters];
        RandomUtils.genTypeWeights(2, weights);

        if (useBiases) {
            this.biases = new double[numFilters][outputHeight][outputWidth];
            RandomUtils.genTypeWeights(2, this.biases);
        }

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


        inputHeight = config[config.length - 1];
        inputWidth = config[config.length - 2];
        channels = config[config.length - 3];

        outputHeight = (((inputHeight - kernelSize1 + (2 * paddingH)) / (stride1)) + 1);
        outputWidth = (((inputWidth - kernelSize2 + (2 * paddingW)) / (stride2)) + 1);


        this.weights = new double[kernelSize1][kernelSize2][channels][numFilters];
        RandomUtils.genTypeWeights(2, weights);

    }

    public Conv2D(int[] shape, int numFilter) {


        this.numFilters = numFilter;
        this.kernelSize1 = 3;
        this.kernelSize2 = 3;
        this.stride1 = 1;
        this.stride2 = 1;


        inputHeight = shape[1];
        inputWidth = shape[2];
        channels = shape[0];

        outputHeight = (((inputHeight - kernelSize1 + (2 * paddingH)) / (stride1)) + 1);
        outputWidth = (((inputWidth - kernelSize2 + (2 * paddingW)) / (stride2)) + 1);


        this.weights = new double[kernelSize1][kernelSize2][channels][this.numFilters];
        RandomUtils.genTypeWeights(2, weights);

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


    @Override
    public void setIterationAt(int iterationAt) {
        this.iterationAt = iterationAt;
    }

    @Override
    public Matrix getOutput() {
        return output;
    }

    @Override
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public int[] getInputShape() {
        return new int[]{channels, inputHeight, inputWidth};
    }

    public int[] getOutputShape() {
        return new int[]{numFilters, outputHeight, outputWidth};
    }


    @Override
    public void setDropout(double rate) {
        this.dropout = new Dropout(rate);
    }

    @Override
    public void setDropout(int size) {
        this.dropout = new Dropout(size);
    }

    @Override
    public void setActivation(Activation act) {
        this.act = act;
    }

    @Override
    public void setUseBiases(boolean useBiases) {
        this.useBiases = useBiases;
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

    public void setWeights(double[][][][] weights, double[][][] biases) {
        this.weights = weights;
        this.biases = biases;
    }

    @Override
    public void setTraining(boolean training) {
        this.training = training;
    }

    @Override
    public Layer getNextLayer() {
        return this.nextLayer;
    }

    @Override
    public void setNextLayer(Layer l) {
        this.nextLayer = l;
    }

    @Override
    public Layer getPreviousLayer() {
        return this.previousLayer;
    }

    @Override
    public void setPreviousLayer(Layer l) {
        this.previousLayer = l;
    }


    @Override
    public void setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
    }

    public Matrix getWeights() {

        if (!this.useBiases) {

            return new Matrix(weights);
        } else {

            double[][][][] result = Arrays.copyOf(weights, weights.length + 1);
            result[weights.length] = this.biases;
            return new Matrix(result);


        }
    }

    public void setWeights(double[][][][] weights) {
        this.weights = weights;
    }

    @Override
    public void setWeights(Matrix m) {

        if (!useBiases) {
            this.weights = m.getData4D();
        } else {
            double[][][][] result = m.getData4D();
            biases = result[result.length - 1];
            weights = Arrays.copyOf(result, result.length - 1);

        }

    }


    public void activateBias() {
        this.useBiases = true;
        this.biases = new double[numFilters][outputHeight][outputWidth];
        RandomUtils.genTypeWeights(2, biases);
    }

    @Override
    public int parameters() {
        return kernelSize1 * kernelSize1 * channels * numFilters;
    }

    public void forward(double[][][] input) {
        this.input = input;

        double[][][] output = new double[numFilters][][];

        for (int m = 0; m < input.length; m++) {
            for (int fi = 0; fi < this.numFilters; fi++) {
                output[fi] = convolveForward(input[m], weights, stride1, m, fi);
            }
        }

        if (this.nextLayer != null) {
            this.nextLayer.forward(new Matrix(output));
        } else {
            this.output = new Matrix(output);
        }

    }

    public void forward(double[][][][] inputs) {
        this.inputs = inputs;

        double[][][][] output = new double[inputs.length][numFilters][][];

        for (int bi = 0; bi < inputs.length; bi++) {
            for (int m = 0; m < inputs[0].length; m++) {
                for (int fi = 0; fi < this.numFilters; fi++) {
                    output[bi][fi] = convolveForward(inputs[bi][m], weights, stride1, m, fi);
                }


            }
        }


        if (this.nextLayer != null) {
            this.nextLayer.forward(new Matrix(output));
        } else {
            this.output = new Matrix(output);
        }

    }

    public void backward(double[] input, double learningRate) {
        this.learningRate = learningRate;
        double[][][] c = reFlat(input, getOutputShape());
        backward(c);

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
                        int inputColIndex = j + y;

                        double value = filter[x][y][m][fi] * input[inputRowIndex][inputColIndex];
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


    public void backward(double[][][] gradInput) {


        double[][][][] filtersDelta = new double[kernelSize1][kernelSize1][channels][numFilters];

        double[][][] dLdOPreviousLayer = new double[channels][inputHeight][inputWidth];


        for (int i = 0; i < input.length; i++) {

            double[][] errorForInput = new double[inputHeight][inputWidth];

            for (int f = 0; f < numFilters; f++) {

                double[][] currFilter = new double[kernelSize1][kernelSize1];

                double[][] error = gradInput[f];

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

        if (this.getPreviousLayer() != null) {
            this.getPreviousLayer().backward(new Matrix<>(dLdOPreviousLayer));
        }
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

    public double[][][][] backward(double[][][][] gradInput, double learningRate) {

        this.learningRate = learningRate;
        for (int i = 0; i < gradInput.length; i++) {
            this.input = inputs[i];
            this.backward(gradInput[i]);
        }

        return gradInput;
    }

    public double[][][][] backward(double[][][][] gradInput) {


        for (int i = 0; i < gradInput.length; i++) {
            this.input = inputs[i];
            this.backward(gradInput[i]);
        }

        return gradInput;
    }


    @Override
    public String summary() {
        return "Conv2D " + "filters: " + numFilters + " inputSize: " + Arrays.toString(getInputShape())
                + " outputSize: " + Arrays.toString(getOutputShape())
                + " parameterSize: " + parameters() + "\n";
    }

    @Override
    public String export() {

        String s = "conv2d;" + useBiases + ";" + this.numFilters + ";" +
                kernelSize1 + ";" + kernelSize2 + ";" + stride1 + ";" + stride2 + ";" + channels + ";" + inputHeight + ";" + inputWidth + "\n";


        s += writeWeights(weights);

        if (useBiases) {
            s += "\n";
            s += writeWeights(biases);
        }
        return s;

    }

    @Override
    public boolean isEqual(Layer other2) {

        Conv2D other = (Conv2D) other2;
        if (Arrays.equals(other.getInputShape(), this.inputShape) && other.stride1 == this.stride1 && other.stride2 == this.stride2
                && other.kernelSize1 == this.kernelSize1 && other.kernelSize2 == this.kernelSize2 && this.numFilters == other.numFilters && this.getWeights().isEquals(other.getWeights())) {
            return true;
        }

        return false;


    }

    public boolean isEqual(Conv2D other) {

        if (other.getInputShape() == this.inputShape && other.stride1 == this.stride1 && other.stride2 == this.stride2
                && other.kernelSize1 == this.kernelSize1 && other.kernelSize2 == this.kernelSize2 && this.numFilters == other.numFilters && this.getWeights() == other.getWeights()) {
            return true;
        }

        return false;


    }

    @Override
    public void forward(Matrix m) {
        int dim = m.getDim();


        if (dim == 3) {
            this.forward(m.getData3D());
        } else if (dim == 4) {
            this.forward(m.getData4D());
        } else {
            System.out.println("Got unsupported Dimension: " + dim);
        }
    }

    @Override
    public void backward(Matrix m) {
        int dim = m.getDim();

        double[][][] c = m.getData3D();
        //printShape(c);
        if (dim == 3) {
            this.backward(m.getData3D());
        } else if (dim == 4) {
            this.backward(m.getData4D());
        } else {
            System.out.println("Got unsupported Dimension: " + dim);
        }
    }

    @Override
    public void backward(Matrix m, double learningRate) {

        if (this.previousLayer != null) {
            this.previousLayer.setLearningRate(learningRate);
        }
        this.learningRate = learningRate;
        int dim = m.getDim();
        if (dim == 3) {
            this.backward(m.getData3D(), learningRate);
        } else if (dim == 4) {
            this.backward(m.getData4D(), learningRate);
        } else {
            System.out.println("Got unsupported Dimension: " + dim);
        }
    }
}
