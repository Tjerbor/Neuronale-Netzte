package layer;

public class MaxPooling2D implements Layer {

    double[][] input;
    double[][][] inputs;
    private int stride;
    private int poolHeights;
    private int poolWidth;

    public MaxPooling2D(int stride, int heights, int width) {
        this.stride = stride;
        this.poolHeights = heights;
        this.poolWidth = width;
    }

    @Override
    public double[][] getWeights() {
        return null;
    }

    @Override
    public void setWeights(double[][] weights) {

    }

    @Override
    public double[][] getMomentumWeights() {
        return new double[0][];
    }

    @Override
    public double[][] getDeltaWeights() {
        return new double[0][];
    }

    @Override
    public double[] getBiases() {
        return new double[0];
    }

    @Override
    public double[] getDeltaBiases() {
        return new double[0];
    }

    @Override
    public double[] getMomentumBiases() {
        return new double[0];
    }

    @Override
    public int parameters() {
        return 0;
    }

    @Override
    public double[] forward(double[] input) {
        return new double[0];
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

    public void setSubmatrix(double[][] in, int h_st, int h_end, int w_st, int w_end, double[][] tmp) {


        for (int i = 0; i < h_end - h_st; i++) {
            for (int j = 0; j < w_end - w_st; j++) {
                in[h_st + i][w_st + j] = tmp[h_st + i][w_st + j];
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

    public double[][] forward(double[][] input) {

        int H = input.length;
        int W = input[0].length;
        int h_out = (1 + (H - poolHeights) / this.stride);
        int w_out = (1 + (W - poolWidth) / this.stride);

        this.input = input;
        double[][] out = new double[h_out][w_out];

        double val;
        double[][] tmp;
        for (int hi = 0; hi < h_out; hi++) {
            for (int wi = 0; wi < w_out; wi++) {
                tmp = getSubmatrix(input, hi * stride, hi * stride + poolHeights, wi * stride, wi * stride + poolWidth);
                val = getMaxValue(tmp);
                out[hi][wi] = val;
            }

        }

        return out;
    }

    @Override
    public double[] backward(double[] input) {
        return new double[0];
    }

    @Override
    public double[] backward(double[] input, double learningRate) {
        return new double[0];
    }


    public double[][][] forward(double[][][] inputs) {
        int H = inputs[0].length;
        int W = inputs[0][0].length;
        int h_out = (1 + (H - poolHeights) / this.stride);
        int w_out = (1 + (W - poolWidth) / this.stride);

        this.inputs = inputs;
        double[][][] out = new double[inputs.length][h_out][w_out];

        double val;
        double[][] tmp;

        for (int i = 0; i < inputs.length; i++) {
            for (int hi = 0; hi < h_out; hi++) {
                for (int wi = 0; wi < w_out; wi++) {
                    tmp = getSubmatrix(inputs[i], hi * stride, hi * stride + poolHeights, wi * stride, wi * stride + poolWidth);
                    val = getMaxValue(tmp);
                    out[i][hi][wi] = val;
                }

            }
        }
        return out;
    }

    public double[][] backward(double[][] delta_inputs) {

        /**
         * needs to find the maximum and the rets is set to zero.
         */
        int H = inputs.length;
        int W = inputs[0].length;
        int h_out = (1 + (H - poolHeights) / this.stride);
        int w_out = (1 + (W - poolWidth) / this.stride);

        double[][] out = new double[this.input.length][this.input[0].length];


        double val;
        double[][] tmp;
        for (int hi = 0; hi < h_out; hi++) {
            for (int wi = 0; wi < w_out; wi++) {
                tmp = getSubmatrixBackward(this.input, hi * stride, hi * stride + poolHeights, wi * stride, wi * stride + poolWidth);
                this.setSubmatrix(out, hi * stride, hi * stride + poolHeights, wi * stride, wi * stride + poolWidth, tmp);
            }

        }

        return out;
    }

    @Override
    public double[][] backward(double[][] inputs, double learningRate) {
        return new double[0][];
    }


    public double[][][] backward(double[][][] delta_inputs) {

        /**
         * needs to find the maximum and the rets is set to zero.
         */
        int H = inputs[0].length;
        int W = inputs[0][0].length;
        int h_out = (1 + (H - poolHeights) / this.stride);
        int w_out = (1 + (W - poolWidth) / this.stride);

        double[][][] out = new double[this.inputs.length][this.inputs[0].length][this.inputs[0][0].length];


        double val;
        double[][] tmp;
        for (int i = 0; i < this.inputs.length; i++) {
            for (int hi = 0; hi < h_out; hi++) {
                for (int wi = 0; wi < w_out; wi++) {
                    tmp = getSubmatrixBackward(this.inputs[i], hi * stride, hi * stride + poolHeights, wi * stride, wi * stride + poolWidth);
                    this.setSubmatrix(out[i], hi * stride, hi * stride + poolHeights, wi * stride, wi * stride + poolWidth, tmp);
                }

            }
        }

        return out;
    }


}
