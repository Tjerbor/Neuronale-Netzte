package layer;


import function.Activation;
import optimizer.Optimizer;
import utils.Array_utils;
import utils.Matrix;
import utils.RandomUtils;
import utils.Utils;

import java.util.Arrays;

import static load.writeUtils.writeShape;
import static load.writeUtils.writeWeights;
import static utils.Array_utils.getShape;
import static utils.Array_utils.sumUpMult;

/**
 * Expect as shape (28, 28,  1)
 * Outputs has Shape (outputHeight, outputWidth,  NUM_FILTERS)
 */

public class Conv2D_Last extends Layer {

    Optimizer optimizer;


    double[][][][] inputs;
    double[][][] input;


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


        this.inputShape = new int[]{inputHeight, inputWidth, channels};
        this.outputShape = new int[]{outputHeight, outputWidth, numFilter};

        this.weights = new double[kernelSize1][kernelSize2][channels][numFilter];
        RandomUtils.genGaussianRandomWeight(weights);

    }

    public Conv2D_Last(int[] shape, int numFilter) {

        this.numFilter = numFilter;
        this.weights = new double[kernelSize1][kernelSize2][shape[2]][numFilter];

        RandomUtils.genGaussianRandomWeight(weights);
        inputHeight = shape[0];
        inputWidth = shape[1];
        channels = shape[2];

        biases = new double[numFilter];
        RandomUtils.genRandomWeight(biases);


        outputHeight = (((inputHeight - kernelSize1 + (2 * paddingH)) / (stride1)) + 1);
        outputWidth = (((inputWidth - kernelSize2 + (2 * paddingW)) / (stride2)) + 1);


        this.inputShape = new int[]{inputHeight, inputWidth, channels};
        this.outputShape = new int[]{outputHeight, outputWidth, numFilter};


    }

    public Conv2D_Last(int[] shape, int numFilter, int[] kernelSize, int[] strides) {

        this.numFilter = numFilter;


        inputHeight = shape[0];
        inputWidth = shape[1];

        this.kernelSize1 = kernelSize[0];
        this.kernelSize2 = kernelSize[1];

        this.stride1 = strides[0];
        this.stride2 = strides[1];


        channels = shape[2];

        outputHeight = (((inputHeight - kernelSize1 + (2 * paddingH)) / (stride1)) + 1);
        outputWidth = (((inputWidth - kernelSize2 + (2 * paddingW)) / (stride2)) + 1);


        this.inputShape = new int[]{inputHeight, inputWidth, channels};
        this.outputShape = new int[]{outputHeight, outputWidth, numFilter};

        if (useBiases) {
            biases = new double[numFilter];
            RandomUtils.genRandomWeight(biases);
        }

        this.weights = new double[kernelSize1][kernelSize2][shape[2]][numFilter];
        RandomUtils.genGaussianRandomWeight(weights);


    }

    public Conv2D_Last(int[] shape, int numFilter, int kernelSize) {


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

        this.inputShape = new int[]{inputHeight, inputWidth, channels};
        this.outputShape = new int[]{outputHeight, outputWidth, numFilter};


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

        this.inputShape = new int[]{inputHeight, inputWidth, channels};
        this.outputShape = new int[]{outputHeight, outputWidth, numFilter};

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
        Conv2D_Last cN = new Conv2D_Last(new int[]{6, 6, channels}, numFilter);

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

        cN.forward(data);

        double[][][][] ist = cN.getOutput().getData4D();
        double[][][][] backIst;


        backIst = cN.backward(ist, 1e-4);

        System.out.println("bakIst: " + Arrays.deepToString(backIst));
        System.out.println(Arrays.toString(getShape(backIst)));


    }

    public void activateBias() {

        this.useBiases = true;
        if (this.biases == null) {
            this.biases = new double[numFilter];
            RandomUtils.genTypeWeights(2, biases);
        }


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


    public void backward(double[] input, double learningRate) {
        this.learningRate = learningRate;
        this.backward(Array_utils.reFlat(input, getInputShape()));

    }


    public void backward(double[][] inputs, double learningRate) {
        this.learningRate = learningRate;
        this.backward(Array_utils.reFlat(inputs, getInputShape()));
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

    @Override
    public Matrix getWeights() {

        if (!useBiases) {
            return new Matrix(weights);
        } else {
            double[][][][] tmp = Arrays.copyOf(weights, weights.length + 1);
            tmp[weights.length] = new double[1][1][biases.length];
            tmp[weights.length][0][0] = biases;
            return new Matrix(tmp);
        }

    }

    @Override
    public void setWeights(Matrix m) {

        if (!useBiases) {
            weights = m.getData4D();
        } else {
            double[][][][] tmp = m.getData4D();
            this.weights = Arrays.copyOf(tmp, tmp.length - 1);
            this.biases = tmp[tmp.length - 1][0][0];
        }
    }

    public void setWeights(double[][][][] weights) {
        this.weights = weights;
    }

    public int[] getInputShape() {
        return new int[]{inputHeight, inputWidth, channels};
    }

    public void setWeights(double[][][][] w, double[] b) {
        this.biases = b;
        this.weights = w;
    }


    @Override
    public void setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
    }

    @Override
    public int parameters() {
        if (useBiases) {
            return biases.length + sumUpMult(getShape(weights));

        } else {
            return sumUpMult(getShape(weights));
        }
    }

    public void genWeights(int type) {

        RandomUtils.genTypeWeights(type, weights);
        if (useBiases) {
            RandomUtils.genTypeWeights(type, biases);
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

    public void sumUpAndUpdateBiases(double[][][][] a) {


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


    public void sumUpAndUpdateBiases(double[][][] a) {


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


    public void forward(double[][][][] inputs) {

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

        this.output = new Matrix(output);
        if (this.getNextLayer() != null) {
            this.getNextLayer().forward(new Matrix(output));
        }
    }

    public void forward(double[][][] input) {

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

        this.output = new Matrix(output);

        if (this.getNextLayer() != null) {
            this.getNextLayer().forward(new Matrix(output));
        }
    }


    public void backward(double[][][][] gradInput) {

        this.sumUpAndUpdateBiases(gradInput);
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
            optimizer.updateParameter(weights, dW);
        } else {
            Utils.updateParameter(weights, dW, this.learningRate);

        }

        if (this.getPreviousLayer() != null) {
            this.getPreviousLayer().backward(new Matrix(dX));
        }
    }


    @Override
    public void setIterationAt(int iterationAt) {
        this.iterationAt = iterationAt;
    }

    @Override
    public Matrix getOutput() {

        if (this.output == null) {
            return null;
        } else {
            return this.output;
        }

    }

    public double[][][][] backward(double[][][][] gradInput, double learningRate) {


        this.learningRate = learningRate;
        this.sumUpAndUpdateBiases(gradInput);

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
            optimizer.setEpochAt(this.iterationAt);
            optimizer.updateParameter(weights, dW);
        } else {
            Utils.updateParameter(weights, dW, learningRate);
        }


        //1250541

        //279513, 86544
        return dX;
    }

    public void backward(double[][][] gradInput, double learningRate) {

        this.learningRate = learningRate;
        this.nextLayer.setLearningRate(learningRate);
        this.backward(gradInput);
    }


    public void backward(double[][][] gradInput) {


        this.sumUpAndUpdateBiases(gradInput);

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
            optimizer.updateParameter(weights, dW);
        } else {
            Utils.updateParameter(weights, dW, learningRate);
        }


        if (this.getPreviousLayer() != null) {
            this.getPreviousLayer().backward(new Matrix(dX));
        }
    }

    public void printOutputShape() {
        System.out.println("Output Shape: " + Arrays.toString(new int[]{this.outputHeight, this.outputWidth, this.numFilter}));

    }

    @Override
    public String export() {

        String s = "conv2d_last;" + useBiases + ";" + numFilter + ";" +
                kernelSize1 + ";" + kernelSize2 + ";" + stride1 + ";" + stride2 + ";" + inputHeight + ";" + inputWidth + ";" + channels + "\n";


        s += writeWeights(weights);

        if (useBiases) {
            s += "\n";
            s += writeWeights(biases);
        }

        return s;
    }

    @Override
    public void setActivation(Activation act) {
        this.act = act;
    }

    public void setUseBiases(boolean useBiases) {
        this.useBiases = useBiases;
    }

    @Override
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }


    public String getConfig() {
        String s = "";


        s += "Kernels: (" + kernelSize1 + ", " + kernelSize1 + ")" + "\n";
        s += "Strides: (" + stride1 + ", " + stride2 + ")" + "\n";
        s += "Num. Filter: " + this.numFilter + "\n";
        s += "inputSize: " + Arrays.toString(getInputShape()) + "\n"
                + "outputSize: " + Arrays.toString(getOutputShape()) + "\n"
                + "useBias: " + useBiases + "\n";

        return s;


    }

    public void printConfig() {
        System.out.println(getConfig());
    }

    @Override
    public String summary() {
        return "Conv2D_Last inputSize: " + Arrays.toString(getInputShape())
                + " outputSize: " + Arrays.toString(getOutputShape())
                + " parameterSize: " + parameters() + "\n";
    }

    @Override
    public boolean isEqual(Layer other2) {

        Conv2D_Last other = (Conv2D_Last) other2;
        if (Arrays.equals(other.getInputShape(), this.inputShape) && other.stride1 == this.stride1 && other.stride2 == this.stride2
                && other.kernelSize1 == this.kernelSize1 && other.kernelSize2 == this.kernelSize2 && this.numFilter == other.numFilter &&
                this.getWeights().isEquals(other.getWeights())) {
            return true;
        }

        if (other.getInputShape() != this.inputShape) {
            System.out.println("inputShape was different: this: " + writeShape(inputShape) + " other: " + writeShape(other.getInputShape()));

        }

        if (other.kernelSize1 != this.kernelSize1 || other.kernelSize2 != this.kernelSize2) {
            System.out.println("KernelSize was different: this: " + kernelSize2 + " : " + kernelSize1);
        }

        if (!(this.getWeights().isEquals(other.getWeights()))) {
            System.out.println("Weights was different");
        }

        return false;


    }

    public boolean isEqual(Conv2D_Last other) {

        if (other.getInputShape() == this.inputShape && other.stride1 == this.stride1 && other.stride2 == this.stride2
                && other.kernelSize1 == this.kernelSize1 && other.kernelSize2 == this.kernelSize2 && this.numFilter == other.numFilter && this.getWeights() == other.getWeights()) {
            return true;
        }

        return false;


    }

}

