package layer;


import utils.Array_utils;
import utils.Matrix;

import java.util.Arrays;

import static load.writeUtils.writeShape;

/**
 * Excpets the deepth of the imgae being one the frist dimension.
 * Image: 1 ´,28, 28
 */
public class MeanPooling2D extends Layer {


    double[][][][] inputs;
    double[][][] input;
    double[][][] outBackward;
    double[][][][] outBackwards;

    boolean Training = true;

    int stride1 = 2;
    int stride2 = 2;
    int kernelSize1 = 2;
    int kernelSize2 = 2;

    int inputH;
    int inputW;
    int channels;

    int out_W;
    int out_H;


    public MeanPooling2D(int[] shape, int[] poolSize, int[] strides) {
        this.stride1 = strides[0];
        this.stride2 = strides[1];

        this.kernelSize1 = poolSize[0];
        this.kernelSize2 = poolSize[1];

        this.channels = shape[0];
        this.inputH = shape[1];
        this.inputW = shape[2];

        out_H = (1 + (inputH - kernelSize1) / stride1);
        out_W = (1 + (inputW - kernelSize2) / stride2);


    }

    public MeanPooling2D(int[] shape, int poolSize) {
        this.kernelSize1 = poolSize;
        this.kernelSize2 = poolSize;

        this.channels = shape[0];
        this.inputH = shape[1];
        this.inputW = shape[2];

        out_H = (1 + (inputH - kernelSize1) / stride1);
        out_W = (1 + (inputW - kernelSize2) / stride1);


    }

    public MeanPooling2D(int[] shape, int poolSize, int stride) {
        this.stride1 = stride;
        this.kernelSize1 = poolSize;
        this.kernelSize2 = poolSize;

        this.channels = shape[0];
        this.inputH = shape[1];
        this.inputW = shape[2];

        out_H = (1 + (inputH - kernelSize1) / stride);
        out_W = (1 + (inputW - kernelSize2) / stride);


    }

    public MeanPooling2D(int[] shape) {
        this.channels = shape[0];
        this.inputH = shape[1];
        this.inputW = shape[2];

        out_H = (1 + (inputH - kernelSize1) / stride1);
        out_W = (1 + (inputW - kernelSize2) / stride1);


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

    public static double[][][][] reshapeImgBack(double[][][][] a) {

        double[][][][] c = new double[a.length][][][];


        for (int i = 0; i < a.length; i++) {
            c[i] = reshapeImgBack(a[i]);
        }

        return c;
    }

    public static double[][][] reshapeImgBack(double[][][] a) {

        double[][][] c = new double[a[0].length][a[0][0].length][a.length];

        for (int ci = 0; ci < a.length; ci++) {
            for (int hi = 0; hi < a[0].length; hi++) {
                for (int wi = 0; wi < a[0][0].length; wi++) {

                    c[hi][wi][ci] = a[ci][hi][wi];
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

    public double[][] getSubmatrix(double[][] in, int h_st, int h_end, int w_st, int w_end) {

        double[][] out = new double[h_end - h_st][w_end - w_st];

        for (int i = 0; i < h_end - h_st; i++) {
            for (int j = 0; j < w_end - w_st; j++) {
                out[i][j] = in[h_st + i][w_st + j];
            }
        }
        return out;
    }


    public double getSubmatrixMean(double[][] in, int h_st, int h_end, int w_st, int w_end) {

        double val = 0;

        for (int i = 0; i < h_end - h_st; i++) {
            for (int j = 0; j < w_end - w_st; j++) {

                val += in[h_st + i][w_st + j];


            }
        }

        return val / kernelSize1 * kernelSize2;
    }

    public double[] getSubmatrixMaxAndIndex(double[][] in, int h_st, int h_end, int w_st, int w_end) {

        int indexOfMaxValueH = 0;
        int indexOfMaxValueW = 0;
        double val = Integer.MIN_VALUE;
        for (int i = 0; i < h_end - h_st; i++) {
            for (int j = 0; j < w_end - w_st; j++) {
                if (val < in[h_st + i][w_st + j]) {
                    val = in[h_st + i][w_st + j];
                    indexOfMaxValueH = h_st + i;
                    indexOfMaxValueW = w_st + j;


                }
            }
        }
        return new double[]{val, indexOfMaxValueH, indexOfMaxValueW};
    }

    public void setSubmatrix(double[][] in, int h_st, int h_end, int w_st, int w_end, double[][] tmp) {


        for (int i = 0; i < h_end - h_st; i++) {
            for (int j = 0; j < w_end - w_st; j++) {
                in[h_st + i][w_st + j] = tmp[i][j];
            }
        }
    }

    public double[][] getSubmatrixBackward(double[][] in, int h_st, int h_end, int w_st, int w_end) {

        double[][] out = new double[h_end - h_st][w_end - w_st];

        int pos1 = 0;
        int pos2 = 0;
        double val = -999999999;
        for (int i = 0; i < h_end - h_st; i++) {
            for (int j = 0; j < w_end - w_st; j++) {
                if (val < in[h_st + i][w_st + j]) {
                    val = in[h_st + i][w_st + j];
                    pos1 = i;
                    pos2 = j;
                }
            }


        }

        out[pos1][pos2] = val;
        return out;
    }

    public double getMaxValue(double[][] in) {

        double val = -99999999;
        for (int i = 0; i < in.length; i++) {
            for (int j = 0; j < in[0].length; j++) {
                if (val < in[i][j]) {
                    val = in[i][j];
                }
            }
        }
        return val;
    }

    public double[][][] forward(double[][][] input) {


        int C = input.length;
        int W = input[0].length;
        int H = input[0][0].length;

        int h_out = (1 + (H - kernelSize1) / stride1);
        int w_out = (1 + (W - kernelSize2) / stride1);

        double[][][] out = new double[C][h_out][w_out];

        this.input = Array_utils.copyArray(input);
        for (int i = 0; i < C; i++) {
            for (int hi = 0; hi < h_out; hi++) {
                for (int wi = 0; wi < w_out; wi++) {
                    out[i][hi][wi] = getSubmatrixMean(input[i], hi * stride1, hi * stride1 + kernelSize1, wi * stride1, wi * stride1 + kernelSize2);
                }

            }

        }

        return out;

    }

    public double[][][][] forward(double[][][][] inputs) {

        int B = inputs.length;
        int C = inputs[0].length;
        int H = inputs[0][0].length;
        int W = inputs[0][0][0].length;

        int H_out = (1 + (H - kernelSize1) / stride1);
        int W_out = (1 + (W - kernelSize2) / stride1);

        double[][][][] out = new double[inputs.length][H_out][W_out][C];

        this.outBackwards = new double[B][C][H][W];

        for (int bs = 0; bs < B; bs++) {
            for (int ci = 0; ci < C; ci++) { //channels
                for (int hi = 0; hi < H_out; hi++) {
                    for (int wi = 0; wi < W_out; wi++) {
                        out[bs][ci][hi][wi] = getSubmatrixMean(inputs[bs][ci], hi * stride1, hi * stride1 + kernelSize1, wi * stride1, wi * stride1 + kernelSize2);
                    }

                }
            }


        }


        return out;

    }


    public double[][][] backwardOld(double[][][] delta_inputs) {


        if (Training) {
            return this.outBackward;
        }

        /**
         * needs to find the maximum and the rets is set to zero.
         */
        int C = input.length;
        int H = input[0].length;
        int W = input[0][0].length;

        int h_out = (1 + (H - kernelSize1) / this.stride1);
        int w_out = (1 + (W - kernelSize2) / this.stride1);

        double[][][] out = new double[C][H][W];

        for (int ci = 0; ci < C; ci++) {
            double[][] tmp;
            for (int hi = 0; hi < h_out; hi++) {
                for (int wi = 0; wi < w_out; wi++) {
                    tmp = getSubmatrixBackward(this.input[ci], hi * stride1, hi * stride1 + kernelSize1, wi * stride1, wi * stride1 + kernelSize2);
                    this.setSubmatrix(out[ci], hi * stride1, hi * stride1 + kernelSize1, wi * stride1, wi * stride1 + kernelSize2, tmp);
                }

            }
        }

        return reshapeImgBack(out);
    }


    public double[][][][] backwardOld(double[][][][] delta_inputs) {

        /**
         * needs to find the maximum and the rets is set to zero.
         */

        int B = inputs.length;
        int C = inputs[0].length;
        int H = inputs[0][0].length;
        int W = inputs[0][0][0].length;

        int h_out = (1 + (H - kernelSize1) / this.stride1);
        int w_out = (1 + (W - kernelSize2) / this.stride1);

        double[][][][] out = new double[B][C][H][W];


        double val;
        double[][] tmp;
        for (int bi = 0; bi < B; bi++) {


            for (int ci = 0; ci < C; ci++) {

                for (int hi = 0; hi < h_out; hi++) {
                    for (int wi = 0; wi < w_out; wi++) {
                        tmp = getSubmatrixBackward(this.inputs[bi][ci], hi * stride1, hi * stride1 + kernelSize1, wi * stride1, wi * stride1 + kernelSize2);
                        this.setSubmatrix(out[bi][ci], hi * stride1, hi * stride1 + kernelSize1, wi * stride1, wi * stride1 + kernelSize2, tmp);
                    }

                }
            }
        }

        return reshapeImgBack(out);
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
            out = new Matrix(this.backwardOld(m.getData3D()));
        } else if (dim == 4) {
            out = new Matrix(this.backwardOld(m.getData4D()));
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
            out = new Matrix(this.backwardOld(m.getData3D()));
        } else if (dim == 4) {
            out = new Matrix(this.backwardOld(m.getData4D()));
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
        return "meanpooling2d;" + kernelSize1 + ";" + kernelSize1 + ";" + stride1 + ";" + stride2 + ";" + writeShape(getInputShape());
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
        return "MeanPooling2D inputSize: " + Arrays.toString(getInputShape())
                + " outputSize: " + Arrays.toString(getOutputShape())
                + " parameterSize: " + parameters() + "\n";
    }
}
