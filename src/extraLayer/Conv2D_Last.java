package extraLayer;


import layer.Activation;
import optimizer.Optimizer;
import utils.Array_utils;
import utils.RandomUtils;
import utils.Utils;

import java.util.Arrays;
import java.util.Random;

import static utils.Array_utils.getShape;

/**
 * Expect as shape (28, 28,  1)
 * Outputs has Shape (outputHeight, outputWidth,  NUM_FILTERS)
 */

public class Conv2D_Last {


    boolean UseMomentum;
    boolean useBiases;

    double[][][] act_input;
    double[][][][] act_inputs;
    double[][][][] inputs;
    double[][][] input;

    Activation act;
    Optimizer optimizer;

    double[] biases;

    double[][][][] weights;

    int kernelSize1 = 5;
    int kernelSize2 = 5;
    int numFilter;

    int channels = 1;


    int stride1 = 1;
    int stride2 = 1;

    int outputHeight;

    int outputWidth;
    int inputWidth;
    int inputHeight;


    int paddingH = 0;
    int paddingW = 0;


    public Conv2D_Last(int[] config) {


        if (config.length < 7) {
            throw new IllegalArgumentException();
        }

        this.numFilter = config[0];
        this.kernelSize1 = config[1];
        this.kernelSize2 = config[2];
        this.stride1 = config[3];
        this.stride2 = config[4];


        inputHeight = config[config.length - 3];
        inputWidth = config[config.length - 2];
        channels = config[config.length - 1];

        outputHeight = (((inputHeight - kernelSize1 + (2 * paddingH)) / (stride1)) + 1);
        outputWidth = (((inputWidth - kernelSize2 + (2 * paddingW)) / (stride2)) + 1);


        this.weights = new double[kernelSize1][kernelSize2][channels][numFilter];
        RandomUtils.genGaussianRandomWeight(weights);

    }

    public Conv2D_Last(int numFilter, int[] shape) {

        this.numFilter = numFilter;
        this.weights = new double[kernelSize1][kernelSize2][shape[0]][numFilter];

        RandomUtils.genGaussianRandomWeight(weights);
        inputHeight = shape[0];
        inputWidth = shape[1];

        biases = new double[numFilter];
        RandomUtils.genRandomWeight(biases);

        channels = shape[2];

        outputHeight = (((inputHeight - kernelSize1 + (2 * paddingH)) / (stride1)) + 1);
        outputWidth = (((inputWidth - kernelSize2 + (2 * paddingW)) / (stride2)) + 1);


        System.out.println(outputHeight);
        System.out.println(outputWidth);
    }


    public Conv2D_Last(int numFilter, int[] shape, int kernelSize) {


        this.kernelSize1 = kernelSize;
        this.kernelSize2 = kernelSize;

        this.numFilter = numFilter;
        this.weights = new double[kernelSize][kernelSize][shape[0]][numFilter];

        RandomUtils.genRandomWeight(weights);
        inputHeight = shape[0];
        inputWidth = shape[1];

        biases = new double[numFilter];
        RandomUtils.genRandomWeight(biases);

        channels = shape[2];

        outputHeight = (((inputHeight - kernelSize1 + (2 * paddingH)) / (stride1)) + 1);
        outputWidth = (((inputWidth - kernelSize2 + (2 * paddingW)) / (stride2)) + 1);


        System.out.println(outputHeight);
        System.out.println(outputWidth);
    }

    public Conv2D_Last(int numFilter, int[] shape, int kernelSize, int stride) {

        this.numFilter = numFilter;
        this.weights = new double[kernelSize1][kernelSize2][shape[0]][numFilter];

        RandomUtils.genRandomWeightConv(weights);
        //RandomUtils.genGaussianRandomWeight(weights);
        inputHeight = shape[0];
        inputWidth = shape[1];

        biases = new double[numFilter];

        channels = shape[2];

        this.kernelSize1 = kernelSize;
        this.kernelSize2 = kernelSize;

        this.stride1 = stride;
        this.stride2 = stride;


        outputHeight = (((inputHeight - kernelSize1 + (2 * paddingH)) / (stride1)) + 1);
        outputWidth = (((inputWidth - kernelSize2 + (2 * paddingW)) / (stride2)) + 1);


    }

    public static double[] sum_axis_0_1_2(double[][][][] a) {


        double[] c = new double[a[0][0][0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                for (int k = 0; k < a[0][0].length; k++) {
                    for (int l = 0; l < a[0][0][0].length; l++) {
                        c[l] += a[i][j][k][l];

                    }
                }
            }
        }
        return c;

    }

    public static void main(String[] args) {

        int batchSize = 1;
        int numFilter = 8;
        int channels = 3;
        Conv2D_Last cN = new Conv2D_Last(numFilter, new int[]{6, 6, channels});

        double[] biases = new double[numFilter];


        for (int i = 0; i < biases.length; i++) {
            biases[i] = 0;
        }


        double[][][][] w = new double[3][3][channels][numFilter];
        int count = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < channels; k++) {
                    for (int l = 0; l < numFilter; l++) {
                        w[i][j][k][l] = count;
                        count += 1;
                    }
                }
            }
        }


        double[][][][] data = new double[batchSize][6][6][channels];

        count = 0;
        for (int bi = 0; bi < batchSize; bi++) {
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 6; j++) {
                    for (int ci = 0; ci < channels; ci++) {
                        data[bi][i][j][ci] = count;
                        count += 1;
                    }
                }
            }
        }
        cN.biases = biases;
        cN.weights = w;

        double[][][][] ist = cN.forward(data);
        double[][][][] backIst;


        backIst = cN.backward(ist, 1e-4);

        System.out.println("bakIst: " + Arrays.deepToString(backIst));
        System.out.println(Arrays.toString(getShape(backIst)));


    }

    public int[] getInputShape() {
        return new int[]{inputHeight, inputWidth, channels};
    }

    public void setWeights(double[][][][] w, double[] b) {
        this.biases = b;
        this.weights = w;


    }

    public void genWeights(int type) {

        Random rand = new Random();
        if (type == 0) {


            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[0].length; j++) {
                    for (int k = 0; k < weights[0][0].length; k++) {
                        for (int l = 0; l < weights[0][0][0].length; l++) {
                            weights[i][j][k][l] = rand.nextGaussian();
                        }
                    }

                }

            }
            if (useBiases) {

                for (int i = 0; i < biases.length; i++) {
                    biases[i] = rand.nextGaussian();
                }

            }

        } else if (type == 1) {
            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[0].length; j++) {
                    for (int k = 0; k < weights[0][0].length; k++) {
                        for (int l = 0; l < weights[0][0][0].length; l++) {
                            weights[i][j][k][l] = rand.nextGaussian(0, 1);
                        }
                    }

                }

            }


            if (useBiases) {

                for (int i = 0; i < biases.length; i++) {
                    biases[i] = rand.nextGaussian(0, 1);
                }

            }

        } else if (type == 2) {
            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[0].length; j++) {
                    for (int k = 0; k < weights[0][0].length; k++) {
                        for (int l = 0; l < weights[0][0][0].length; l++) {
                            weights[i][j][k][l] = rand.nextDouble(-0.1, 0.1);
                        }
                    }

                }

            }
            if (useBiases) {

                for (int i = 0; i < biases.length; i++) {
                    biases[i] = rand.nextDouble(-0.1, 0.1);
                }

            }
        } else if (type == 3) {
            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[0].length; j++) {
                    for (int k = 0; k < weights[0][0].length; k++) {
                        for (int l = 0; l < weights[0][0][0].length; l++) {
                            weights[j][i][k][l] = rand.nextDouble(-1, 1);
                        }
                    }

                }

            }
            if (useBiases) {

                for (int i = 0; i < biases.length; i++) {
                    biases[i] = rand.nextDouble(-1, 1);
                }

            }
        } else if (type == 4) {
            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[0].length; j++) {
                    for (int k = 0; k < weights[0][0].length; k++) {
                        for (int l = 0; l < weights[0][0][0].length; l++) {
                            weights[i][j][k][l] = rand.nextGaussian();
                        }
                    }

                }

            }
            if (useBiases) {

                for (int i = 0; i < biases.length; i++) {
                    biases[i] = rand.nextGaussian();
                }

            }
        }

    }

    public void setBiases(double[] biases) {
        this.biases = biases;
    }

    public void setStride1(int stride1) {
        this.stride1 = stride1;


        outputHeight = (((inputHeight - kernelSize1 + (2 * paddingH)) / (stride1)) + 1);
        outputWidth = (((inputWidth - kernelSize2 + (2 * paddingW)) / (stride2)) + 1);

    }

    public void setStride2(int stride2) {
        this.stride2 = stride2;

        outputHeight = (((inputHeight - kernelSize1 + (2 * paddingH)) / (stride1)) + 1);
        outputWidth = (((inputWidth - kernelSize2 + (2 * paddingW)) / (stride2)) + 1);

    }

    public void setStrides(int[] strides) {
        this.stride1 = strides[0];
        this.stride2 = strides[1];

        outputHeight = (((inputHeight - kernelSize1 + (2 * paddingH)) / (stride1)) + 1);
        outputWidth = (((inputWidth - kernelSize2 + (2 * paddingW)) / (stride2)) + 1);

    }

    public void setStrides(int stride) {
        this.stride1 = stride;
        this.stride2 = stride;

        outputHeight = (((inputHeight - kernelSize1 + (2 * paddingH)) / (stride1)) + 1);
        outputWidth = (((inputWidth - kernelSize2 + (2 * paddingW)) / (stride2)) + 1);

    }

    public void setKernelSize(int[] kernelSize) {
        this.kernelSize1 = kernelSize[0];
        this.kernelSize2 = kernelSize[0];

        outputHeight = (((inputHeight - kernelSize1 + (2 * paddingH)) / (stride1)) + 1);
        outputWidth = (((inputWidth - kernelSize2 + (2 * paddingW)) / (stride2)) + 1);
    }

    public void setKernelSize(int kernelSize) {
        this.kernelSize1 = kernelSize;
        this.kernelSize2 = kernelSize;

        outputHeight = (((inputHeight - kernelSize1 + (2 * paddingH)) / (stride1)) + 1);
        outputWidth = (((inputWidth - kernelSize2 + (2 * paddingW)) / (stride2)) + 1);
    }

    public void setKernelSize(int kernelSize1, int kernelSize2) {
        this.kernelSize1 = kernelSize1;
        this.kernelSize2 = kernelSize2;

        outputHeight = (((inputHeight - kernelSize1 + (2 * paddingH)) / (stride1)) + 1);
        outputWidth = (((inputWidth - kernelSize2 + (2 * paddingW)) / (stride2)) + 1);
    }

    public void setKernelSize1(int kernelSize1) {
        this.kernelSize1 = kernelSize1;

        outputHeight = (((inputHeight - kernelSize1 + (2 * paddingH)) / (stride1)) + 1);
        outputWidth = (((inputWidth - kernelSize2 + (2 * paddingW)) / (stride2)) + 1);
    }

    public void setKernelSize2(int kernelSize2) {
        this.kernelSize2 = kernelSize2;

        outputHeight = (((inputHeight - kernelSize1 + (2 * paddingH)) / (stride1)) + 1);
        outputWidth = (((inputWidth - kernelSize2 + (2 * paddingW)) / (stride2)) + 1);
    }

    public void sumUpAndUpdateBiases(double[][][][] a, double learningRate) {


        double[] c = new double[a[0][0][0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                for (int k = 0; k < a[0][0].length; k++) {
                    for (int l = 0; l < a[0][0][0].length; l++) {
                        c[l] += a[i][j][k][l];

                    }
                }
            }
        }

        if (optimizer != null) {
            optimizer.updateParameter(this.biases, c);
        } else {

            Utils.updateParameter(this.biases, c, learningRate);
        }

    }

    public void sumUpAndUpdateBiases(double[][][] a, double learningRate) {


        double[] c = new double[a[0][0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                for (int k = 0; k < a[0][0].length; k++) {
                    c[k] += a[i][j][k];


                }
            }
        }

        if (optimizer != null) {
            optimizer.updateParameter(this.biases, c);
        } else {

            Utils.updateParameter(this.biases, c, learningRate);
        }


    }

    public double getMultiSum(int pos, double[][][] in, int h_st, int h_end, int w_st, int w_end) {

        double out = 0;


        for (int ci = 0; ci < in[0][0].length; ci++) {
            for (int i = 0; i < h_end - h_st; i++) {
                for (int j = 0; j < w_end - w_st; j++) {
                    out += in[h_st + i][w_st + j][ci] * weights[i][j][ci][pos];

                }
            }
        }

        if (useBiases) {
            return out + biases[pos];
        } else {
            return out;
        }

    }

    public void getMultiBackward(int pos, double[][][] in, int h_st, int h_end, int w_st, int w_end, double dZScalar, double[][][][] dW) {


        for (int ci = 0; ci < in[0][0].length; ci++) {
            for (int i = 0; i < h_end - h_st; i++) {
                for (int j = 0; j < w_end - w_st; j++) {
                    dW[j][i][ci][pos] += in[h_st + i][w_st + j][ci] * dZScalar;
                }
            }
        }

    }

    public void getOutputGradient(int pos, double[][][] dx, int h_st, int h_end, int w_st, int w_end, double dZScalar) {

        for (int ci = 0; ci < channels; ci++) {
            for (int i = 0; i < h_end - h_st; i++) {
                for (int j = 0; j < w_end - w_st; j++) {
                    dx[h_st + i][w_st + j][ci] = weights[i][j][ci][pos] * dZScalar;
                }
            }
        }


    }

    public double[][][][] forward(double[][][][] inputs) {

        double[][][][] output = new double[inputs.length][outputHeight][outputWidth][numFilter];


        this.inputs = Array_utils.copyArray(inputs);
        double tmp;


        for (int bs = 0; bs < inputs.length; bs++) {

            for (int i = 0; i < outputHeight; i++) {
                for (int j = 0; j < outputWidth; j++) {
                    for (int ci = 0; ci < numFilter; ci++) {

                        tmp = getMultiSum(ci, inputs[bs], i * (stride1), kernelSize1 + (i * (stride1)),
                                j * (stride2), kernelSize2 + (j * (stride2)));

                        output[bs][i][j][ci] = tmp;


                    }
                }

            }
        }


        //1250541

        //279513, 86544


        return output;
    }

    public double[][][] forward(double[][][] input) {

        double[][][] output = new double[outputHeight][outputWidth][numFilter];


        this.input = Array_utils.copyArray(input);
        double tmp;


        for (int i = 0; i < outputHeight; i++) {
            for (int j = 0; j < outputWidth; j++) {
                for (int ci = 0; ci < numFilter; ci++) {
                    tmp = getMultiSum(ci, input, i * (stride1), kernelSize1 + (i * (stride1)),
                            j * (stride2), kernelSize2 + (j * (stride2)));


                    output[i][j][ci] = tmp;
                }
            }


        }


        //1250541

        //279513, 86544


        return output;
    }

    public double[][][][] backward(double[][][][] gradInput, double learningRate) {


        this.sumUpAndUpdateBiases(gradInput, learningRate);

        double[][][][] dW = Array_utils.zerosLike(this.weights);
        double[][][][] dX = Array_utils.zerosLike(this.inputs);


        for (int bs = 0; bs < gradInput.length; bs++) {
            for (int i = 0; i < outputHeight; i++) {
                for (int j = 0; j < outputWidth; j++) {
                    for (int ci = 0; ci < numFilter; ci++) {


                        getMultiBackward(ci, inputs[bs], i * (stride1), kernelSize1 + (i * (stride1)),
                                j * (stride2), kernelSize2 + (j * (stride2)), gradInput[bs][i][j][ci], dW);

                        getOutputGradient(ci, dX[bs], i * (stride1), kernelSize1 + (i * (stride1)),
                                j * (stride2), kernelSize2 + (j * (stride2)), gradInput[bs][i][j][ci]);
                    }
                }

            }
        }


        if (optimizer != null) {
            optimizer.setLearningRate(learningRate);
            optimizer.updateParameter(weights, dW);
        } else {
            Utils.updateParameter(weights, dW, learningRate);
        }

        //


        //1250541
        //279513, 86544
        return dX;
    }

    public double[][][][] backward(double[][][][] gradInput, double learningRate, int t) {


        this.sumUpAndUpdateBiases(gradInput, learningRate);

        double[][][][] dW = Array_utils.zerosLike(this.weights);
        double[][][][] dX = Array_utils.zerosLike(this.inputs);


        for (int bs = 0; bs < gradInput.length; bs++) {
            for (int i = 0; i < outputHeight; i++) {
                for (int j = 0; j < outputWidth; j++) {
                    for (int ci = 0; ci < numFilter; ci++) {


                        getMultiBackward(ci, inputs[bs], i * (stride1), kernelSize1 + (i * (stride1)),
                                j * (stride2), kernelSize2 + (j * (stride2)), gradInput[bs][i][j][ci], dW);

                        getOutputGradient(ci, dX[bs], i * (stride1), kernelSize1 + (i * (stride1)),
                                j * (stride2), kernelSize2 + (j * (stride2)), gradInput[bs][i][j][ci]);
                    }
                }

            }
        }

        if (optimizer != null) {
            optimizer.setLearningRate(learningRate);
            optimizer.updateParameter(weights, dW, t);
        } else {
            Utils.updateParameter(weights, dW, learningRate);
        }


        //1250541

        //279513, 86544
        return dX;
    }

    public double[][][] backward(double[][][] gradInput, double learningRate) {


        this.sumUpAndUpdateBiases(gradInput, learningRate);

        double[][][][] dW = Array_utils.zerosLike(this.weights);
        double[][][] dX = Array_utils.zerosLike(this.input);


        for (int i = 0; i < outputHeight; i++) {
            for (int j = 0; j < outputWidth; j++) {
                for (int ci = 0; ci < numFilter; ci++) {

                    getMultiBackward(ci, input, i * (stride1), kernelSize1 + (i * (stride1)),
                            j * (stride2), kernelSize2 + (j * (stride2)), gradInput[i][j][ci], dW);

                    getOutputGradient(ci, dX, i * (stride1), kernelSize1 + (i * (stride1)),
                            j * (stride2), kernelSize2 + (j * (stride2)), gradInput[i][j][ci]);
                }
            }


        }


        if (optimizer != null) {
            optimizer.setLearningRate(learningRate);
            optimizer.updateParameter(weights, dW);
        } else {
            Utils.updateParameter(weights, dW, learningRate);
        }


        //1250541

        //279513, 86544
        return dX;
    }


    public void printOutputShape() {
        System.out.println("Output Shape: " + Arrays.toString(new int[]{this.outputHeight, this.outputWidth, this.numFilter}));

    }


    public int[] getOutputShape() {
        return new int[]{this.outputHeight, this.outputWidth, this.numFilter};

    }


    public void setWeights(double[][][][] weights) {
        this.weights = weights;
    }

    public void setUseBiases(boolean useBiases) {
        this.useBiases = useBiases;
    }


    public String getConfig() {
        String s = "";


        s += "Kernels: (" + kernelSize1 + ", " + kernelSize1 + ")" + "\n";
        s += "Strides: (" + stride1 + ", " + stride2 + ")" + "\n";
        s += "Num. Filter: " + this.numFilter + "\n";

        return s;


    }

    public void printConfig() {

        System.out.println(getConfig());
    }


}

