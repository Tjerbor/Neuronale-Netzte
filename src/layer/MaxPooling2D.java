package layer;

public class MaxPooling2D {


    double[][][][] inputs;
    double[][][] input;

    int stride = 2;
    int poolHeight = 2;
    int poolWidth = 2;


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


        int H = input.length;
        int W = input[0].length;
        int C = input[0][0].length;

        int h_out = (int) (1 + (H - poolHeight) / stride);
        int w_out = (int) (1 + (W - poolWidth) / stride);

        double[][][] out = new double[h_out][w_out][C];
        double[][] tmp;
        double val;


        input = reshapeImg(input);
        this.input = input;
        for (int i = 0; i < C; i++) {
            for (int hi = 0; hi < h_out; hi++) {
                for (int wi = 0; wi < w_out; wi++) {
                    tmp = getSubmatrix(input[i], hi * stride, hi * stride + poolHeight, wi * stride, wi * stride + poolWidth);
                    val = getMaxValue(tmp);
                    out[wi][hi][i] = val;
                }

            }
        }

        return out;

    }

    public double[][][][] forward(double[][][][] inputs) {


        int H = inputs[0].length;
        int W = inputs[0][0].length;
        int C = inputs[0][0][0].length;

        int H_out = (int) (1 + (H - poolHeight) / stride);
        int W_out = (int) (1 + (W - poolWidth) / stride);

        double[][][][] out = new double[inputs.length][H_out][W_out][C];
        double[][] tmp;

        double val;

        inputs = reshapeImg(inputs);
        this.inputs = inputs;


        for (int bs = 0; bs < inputs.length; bs++) {
            for (int ci = 0; ci < C; ci++) { //channels
                for (int hi = 0; hi < H_out; hi++) {
                    for (int wi = 0; wi < W_out; wi++) {
                        tmp = getSubmatrix(inputs[bs][ci], hi * stride, hi * stride + poolHeight, wi * stride, wi * stride + poolWidth);
                        val = getMaxValue(tmp);
                        out[bs][wi][hi][ci] = val;
                    }

                }

            }
        }


        return out;

    }

    public double[][][] backward(double[][][] delta_inputs) {


        /**
         * needs to find the maximum and the rets is set to zero.
         */
        int C = input.length;
        int H = input[0].length;
        int W = input[0][0].length;

        int h_out = (1 + (H - poolHeight) / this.stride);
        int w_out = (1 + (W - poolWidth) / this.stride);

        double[][][] out = new double[C][H][W];

        for (int ci = 0; ci < C; ci++) {
            double[][] tmp;
            for (int hi = 0; hi < h_out; hi++) {
                for (int wi = 0; wi < w_out; wi++) {
                    tmp = getSubmatrixBackward(this.input[ci], hi * stride, hi * stride + poolHeight, wi * stride, wi * stride + poolWidth);
                    this.setSubmatrix(out[ci], hi * stride, hi * stride + poolHeight, wi * stride, wi * stride + poolWidth, tmp);
                }

            }
        }

        return reshapeImgBack(out);
    }


    public double[][][][] backward(double[][][][] delta_inputs) {

        /**
         * needs to find the maximum and the rets is set to zero.
         */

        int B = inputs.length;
        int C = inputs[0].length;
        int H = inputs[0][0].length;
        int W = inputs[0][0][0].length;

        int h_out = (1 + (H - poolHeight) / this.stride);
        int w_out = (1 + (W - poolWidth) / this.stride);

        double[][][][] out = new double[B][C][H][W];


        double val;
        double[][] tmp;
        for (int bi = 0; bi < B; bi++) {


            for (int ci = 0; ci < C; ci++) {

                for (int hi = 0; hi < h_out; hi++) {
                    for (int wi = 0; wi < w_out; wi++) {
                        tmp = getSubmatrixBackward(this.inputs[bi][ci], hi * stride, hi * stride + poolHeight, wi * stride, wi * stride + poolWidth);
                        this.setSubmatrix(out[bi][ci], hi * stride, hi * stride + poolHeight, wi * stride, wi * stride + poolWidth, tmp);
                    }

                }
            }
        }

        return reshapeImgBack(out);
    }

}
