package layer;

import static utils.Array_utils.getSubmatrix;

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

        this.input = input;
        int H = input.length;
        int W = input[0].length;
        int C = input[0][0].length;

        int h_out = (int) (1 + (H - poolHeight) / stride);
        int w_out = (int) (1 + (W - poolWidth) / stride);

        double[][][] out = new double[h_out][w_out][C];
        double[][] tmp;
        double val;

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

        this.inputs = inputs;
        int H = inputs[0].length;
        int W = inputs[0][0].length;
        int C = inputs[0][0][0].length;

        int H_out = (int) (1 + (H - poolHeight) / stride);
        int W_out = (int) (1 + (W - poolWidth) / stride);

        double[][][][] out = new double[inputs.length][H_out][W_out][C];
        double[][] tmp;

        double val;

        inputs = reshapeImg(inputs);
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

}
