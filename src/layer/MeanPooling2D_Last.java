package layer;

import utils.Array_utils;
import utils.Matrix;

import java.util.Arrays;

import static load.writeUtils.writeShape;

public class MeanPooling2D_Last extends Layer {


    int stride1 = 2;
    int stride2 = 2;
    int kernelSize1 = 2;
    int kernelSize2 = 2;

    int inputHeight;
    int inputWidth;

    int[] inputShape;


    int outputHeight;
    int outputWidth;
    double[][][] input;
    double[][][][] inputs;


    public MeanPooling2D_Last(int[] inputShape) {

        outputHeight = (((inputHeight - kernelSize1) / (stride1)) + 1);
        outputWidth = (((inputWidth - kernelSize2) / (stride2)) + 1);

        this.inputShape = inputShape;
        outputShape = new int[]{outputWidth, outputWidth, inputShape[2]};


    }

    public MeanPooling2D_Last(int[] inputShape, int kernelSize, int stride) {

        outputHeight = (((inputHeight - kernelSize1) / (stride1)) + 1);
        outputWidth = (((inputWidth - kernelSize2) / (stride2)) + 1);

        this.inputShape = inputShape;

        this.kernelSize1 = kernelSize;
        this.kernelSize2 = kernelSize;

        this.stride1 = stride;
        this.stride2 = stride;

        this.inputShape = inputShape;
        outputShape = new int[]{outputWidth, outputWidth, inputShape[2]};


    }

    public MeanPooling2D_Last(int[] inputShape, int[] kernelSize, int[] strides) {

        outputHeight = (((inputHeight - kernelSize1) / (stride1)) + 1);
        outputWidth = (((inputWidth - kernelSize2) / (stride2)) + 1);

        this.inputShape = inputShape;

        this.kernelSize1 = kernelSize[0];
        this.kernelSize2 = kernelSize[1];

        this.stride1 = strides[0];
        this.stride2 = strides[1];

        this.inputShape = inputShape;
        outputShape = new int[]{outputWidth, outputWidth, inputShape[2]};


    }


    public double getSubMatrixMean(int pos, double[][][] in, int h_st, int h_end, int w_st, int w_end) {

        double sum = 0;
        int count = 0;

        for (int i = 0; i < h_end - h_st; i++) {
            for (int j = 0; j < w_end - w_st; j++) {
                sum += in[h_st + i][w_st + j][pos];
                count += 1;

            }
        }


        return sum / count;
    }

    public double getSubMatrixGrad(int pos, double[][][] in, int h_st, int h_end, int w_st, int w_end, double dZ) {

        int sum = 0;


        for (int i = 0; i < h_end - h_st; i++) {
            for (int j = 0; j < w_end - w_st; j++) {
                in[h_st + i][w_st + j][pos] += dZ;


            }
        }


        return sum;
    }

    public double[][][] forward(double[][][] input) {


        int channels = input[0][0].length;
        double[][][] output = new double[outputHeight][outputWidth][channels];

        this.input = Array_utils.copyArray(input);
        this.inputShape = Array_utils.getShape(input);

        double tmp;
        for (int i = 0; i < outputHeight; i++) {
            for (int j = 0; j < outputWidth; j++) {
                for (int ci = 0; ci < channels; ci++) {

                    tmp = getSubMatrixMean(ci, input, i * (stride1), kernelSize1 + (i * (stride1)),
                            j * (stride2), kernelSize2 + (j * (stride2)));

                    output[i][j][ci] = tmp;

                }
            }

        }

        return output;
    }

    public double[][][][] forward(double[][][][] inputs) {


        int channels = inputs[0][0][0].length;
        double[][][][] output = new double[inputs.length][outputHeight][outputWidth][channels];

        this.inputs = Array_utils.copyArray(inputs);

        double tmp;
        for (int bs = 0; bs < inputs.length; bs++) {

            for (int i = 0; i < outputHeight; i++) {
                for (int j = 0; j < outputWidth; j++) {
                    for (int ci = 0; ci < channels; ci++) {

                        tmp = getSubMatrixMean(ci, inputs[bs], i * (stride1), kernelSize1 + (i * (stride1)),
                                j * (stride2), kernelSize2 + (j * (stride2)));

                        output[bs][i][j][ci] = tmp;

                    }
                }

            }
        }

        return output;
    }

    public double[][][] backward(double[][][] gardInput) {


        int channels = input[0][0].length;
        double[][][] output = new double[inputShape[0]][inputShape[1]][inputShape[2]];


        double tmp;
        for (int i = 0; i < outputHeight; i++) {
            for (int j = 0; j < outputWidth; j++) {
                for (int ci = 0; ci < channels; ci++) {
                    tmp = getSubMatrixGrad(ci, output, i * (stride1), kernelSize1 + (i * (stride1)),
                            j * (stride2), kernelSize2 + (j * (stride2)), gardInput[i][j][ci] / ((double) kernelSize1 / kernelSize2));

                    output[i][j][ci] += tmp;

                }
            }

        }


        return output;
    }

    public double[][][][] backward(double[][][][] gardInput) {


        int channels = inputs[0][0][0].length;
        double[][][][] output = Array_utils.zerosLike(this.inputs);


        double tmp;
        for (int bs = 0; bs < inputs.length; bs++) {
            for (int i = 0; i < outputHeight; i++) {
                for (int j = 0; j < outputWidth; j++) {
                    for (int ci = 0; ci < channels; ci++) {
                        tmp = getSubMatrixGrad(ci, output[bs], i * (stride1), kernelSize1 + (i * (stride1)),
                                j * (stride2), kernelSize2 + (j * (stride2)), gardInput[bs][i][j][ci] / ((double) kernelSize1 / kernelSize2));

                        output[bs][i][j][ci] += tmp;

                    }
                }

            }
        }

        return output;
    }


    @Override
    public void forward(Matrix m) {
        int dim = m.getDim();

        Matrix out;
        if (dim == 3) {
            out = new Matrix(this.forward(m.getData3D()));
        } else if (dim == 4) {
            out = new Matrix(this.forward(m.getData4D()));
        } else {
            throw new IllegalArgumentException("Got unsupported Dimension: " + dim);
        }

        if (this.nextLayer != null) {
            this.nextLayer.forward(out);
        } else {
            this.output = out;
        }

    }

    @Override
    public void backward(Matrix m) {
        int dim = m.getDim();
        Matrix out;

        if (dim == 3) {
            out = new Matrix(this.backward(m.getData3D()));
        } else if (dim == 4) {
            out = new Matrix(this.backward(m.getData4D()));
        } else {
            throw new IllegalArgumentException("Got unsupported Dimension: " + dim);
        }
        if (this.previousLayer != null) {
            this.previousLayer.backward(out);
        }

    }

    @Override
    public void backward(Matrix m, double learningRate) {

        if (this.previousLayer != null) {
            this.previousLayer.setLearningRate(learningRate);
        }

        Matrix out;

        this.learningRate = learningRate;
        int dim = m.getDim();
        if (dim == 3) {
            out = new Matrix(this.backward(m.getData3D()));
        } else if (dim == 4) {
            out = new Matrix(this.backward(m.getData4D()));
        } else {
            throw new IllegalArgumentException("Got unsupported Dimension: " + dim);
        }
        if (this.previousLayer != null) {
            this.previousLayer.backward(out, learningRate);
        }
    }

    @Override
    public Matrix getWeights() {
        return null;
    }

    @Override
    public void setWeights(Matrix m) {

    }

    @Override
    public String export() {
        return "meanpooling2d_last;" + kernelSize1 + ";" + kernelSize1 + ";" + stride1 + ";" + stride2 + ";" + writeShape(getInputShape());
    }

    @Override
    public boolean isEqual(Layer other) {

        MaxPooling2D_Last tmp = (MaxPooling2D_Last) other;

        if ((Arrays.equals(tmp.outputShape, this.outputShape) && Arrays.equals(tmp.inputShape, this.inputShape)) && this.kernelSize1 == tmp.kernelSize1 && tmp.kernelSize2 == kernelSize2
                && this.stride1 == tmp.stride1 && this.stride2 == tmp.stride2) {

            return true;
        }


        return false;
    }

    public String summary() {
        return "MeanPooling2D_Last inputSize: " + Arrays.toString(getInputShape())
                + " outputSize: " + Arrays.toString(getOutputShape())
                + " parameters: " + parameters() + "\n";
    }


}
