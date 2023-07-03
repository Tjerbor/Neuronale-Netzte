package layer;


import utils.Matrix;

import java.util.Arrays;

import static load.writeUtils.writeShape;

public class MaxPooling2D extends Layer {

    double[][][][] inputs;
    double[][][] input;
    double[][][] outBackward;
    double[][][][] outBackwards;


    boolean Training = true;

    int stride1 = 2;
    int stride2 = 2;
    int kernelSize1 = 2;
    int kernelSize2 = 2;


    int[][][][] inputMaxIndicies_R;
    int[][][][] inputMaxIndicies_C;

    int inputH;
    int inputW;
    int channels;

    int out_W;
    int out_H;


    public MaxPooling2D(int[] shape, int[] poolSize, int[] strides) {
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

    public MaxPooling2D(int[] shape, int poolSize) {
        this.kernelSize1 = poolSize;
        this.kernelSize2 = poolSize;

        this.channels = shape[0];
        this.inputH = shape[1];
        this.inputW = shape[2];

        out_H = (1 + (inputH - kernelSize1) / stride1);
        out_W = (1 + (inputW - kernelSize2) / stride1);


    }

    public MaxPooling2D(int[] shape, int poolSize, int stride) {
        this.stride1 = stride;
        this.kernelSize1 = poolSize;
        this.kernelSize2 = poolSize;

        this.channels = shape[0];
        this.inputH = shape[1];
        this.inputW = shape[2];

        out_H = (1 + (inputH - kernelSize1) / stride);
        out_W = (1 + (inputW - kernelSize2) / stride);


    }

    public MaxPooling2D(int[] shape) {
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

    public int getOutputRows() {
        int h_out = (1 + (this.inputH - kernelSize1) / stride1);
        return h_out;


    }

    public int getOutputCols() {
        return (1 + (this.inputW - kernelSize2) / stride1);


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


    public double getSubmatrixMax(double[][] in, int h_st, int h_end, int w_st, int w_end) {

        double val = Integer.MIN_VALUE;

        for (int i = 0; i < h_end - h_st; i++) {
            for (int j = 0; j < w_end - w_st; j++) {
                if (val < in[h_st + i][w_st + j]) {
                    val = in[h_st + i][w_st + j];


                }
            }
        }

        return val;
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

    public double[][][][] forward(double[][][][] input) {


        int batchSize = input.length;
        int C = input[0].length;
        int W = input[0][0].length;
        int H = input[0][0][0].length;

        int h_out = (1 + (H - kernelSize1) / stride1);
        int w_out = (1 + (W - kernelSize2) / stride1);

        double[][][][] out = new double[batchSize][C][h_out][w_out];

        inputMaxIndicies_R = new int[batchSize][channels][out_H][out_W];
        inputMaxIndicies_C = new int[batchSize][channels][out_H][out_W];

        this.inputs = input;

        double[][] tmpBack = new double[H][W];

        double[] tmp = new double[3];


        int wb;
        int hb;
        for (int bs = 0; bs < batchSize; bs++) {


            for (int i = 0; i < C; i++) {
                tmpBack = new double[H][W];
                for (int hi = 0; hi < h_out; hi++) {
                    for (int wi = 0; wi < w_out; wi++) {
                        tmp = getSubmatrixMaxAndIndex(input[bs][i], hi * stride1, hi * stride1 + kernelSize1, wi * stride1, wi * stride1 + kernelSize2);
                        out[bs][i][hi][wi] = tmp[0];
                        hb = (int) tmp[1];
                        wb = (int) tmp[2];
                        tmpBack[hb][wb] = tmp[0];
                        inputMaxIndicies_R[bs][i][hi][wi] = hb;
                        inputMaxIndicies_C[bs][i][hi][wi] = wb;


                    }

                }

            }

        }
        return out;

    }

    public double[][][] forward(double[][][] input) {


        int C = input.length;
        int W = input[0].length;
        int H = input[0][0].length;

        int h_out = (1 + (H - kernelSize1) / stride1);
        int w_out = (1 + (W - kernelSize2) / stride1);

        double[][][] out = new double[C][h_out][w_out];

        inputMaxIndicies_R = new int[1][channels][out_H][out_W];
        inputMaxIndicies_C = new int[1][channels][out_H][out_W];

        this.input = input;

        double[][] tmpBack = new double[H][W];

        this.outBackward = new double[C][H][W];
        double[] tmp = new double[3];


        int wb;
        int hb;
        for (int i = 0; i < C; i++) {
            tmpBack = new double[H][W];
            for (int hi = 0; hi < h_out; hi++) {
                for (int wi = 0; wi < w_out; wi++) {
                    tmp = getSubmatrixMaxAndIndex(input[i], hi * stride1, hi * stride1 + kernelSize1, wi * stride1, wi * stride1 + kernelSize2);
                    out[i][hi][wi] = tmp[0];
                    hb = (int) tmp[1];
                    wb = (int) tmp[2];
                    tmpBack[hb][wb] = tmp[0];
                    inputMaxIndicies_R[0][i][hi][wi] = hb;
                    inputMaxIndicies_C[0][i][hi][wi] = wb;


                }

            }

        }

        return out;

    }

    public double[][][] backward(double[][][] dLdO) {

        double[][][] dXdL = new double[channels][][];

        int l = 0;


        for (int i = 0; i < dLdO.length; i++) {
            double[][] error = new double[inputH][inputW];
            for (int r = 0; r < getOutputRows(); r++) {
                for (int c = 0; c < getOutputCols(); c++) {
                    int max_i = inputMaxIndicies_R[0][i][r][c];
                    int max_j = inputMaxIndicies_C[0][i][r][c];

                    if (max_i != -1) {
                        error[max_i][max_j] += dLdO[i][r][c];
                    }
                }
            }
            dXdL[i] = error;
        }

        return dXdL;

    }

    public double[][][][] backward(double[][][][] dLdO) {

        int batchSize = dLdO.length;
        double[][][][] dXdL = new double[batchSize][channels][][];

        int l = 0;

        for (int bs = 0; bs < batchSize; bs++) {
            for (int i = 0; i < dLdO[0].length; i++) {
                double[][] error = new double[inputH][inputW];
                for (int r = 0; r < getOutputRows(); r++) {
                    for (int c = 0; c < getOutputCols(); c++) {
                        int max_i = inputMaxIndicies_R[0][i][r][c];
                        int max_j = inputMaxIndicies_C[0][i][r][c];

                        if (max_i != -1) {
                            error[max_i][max_j] += dLdO[bs][i][r][c];
                        }
                    }
                }
                dXdL[bs][i] = error;
            }

        }
        return dXdL;

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
    public int[] getOutputShape() {
        return new int[]{channels, out_H, out_W};

    }

    @Override
    public int[] getInputShape() {
        return new int[]{channels, inputH, inputW};

    }


    @Override
    public String export() {
        return "maxpooling2d;" + kernelSize1 + ";" + kernelSize2 + ";" + stride1 + ";" + stride2 + ";" + writeShape(getInputShape());
    }

    @Override
    public boolean isEqual(Layer other) {
        return false;
    }

    public String summary() {
        return "MaxPooling2D inputSize: " + Arrays.toString(getInputShape())
                + " outputSize: " + Arrays.toString(getOutputShape())
                + " parameters: " + parameters() + "\n";
    }


}
