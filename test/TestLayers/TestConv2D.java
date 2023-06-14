package TestLayers;

import layer.Conv2D;

import java.util.Arrays;

public class TestConv2D {


    public static void main(String[] args) {
        Conv2D C = new Conv2D(8, new int[]{1, 6, 6}, 3);

        double[][] a = new double[][]{{0, 1, 2,},
                {3, 4, 5},
                {6, 7, 8}};

        double[][] b = new double[6][6];

        int count = 0;
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                b[i][j] = count;
                count += 1;
            }
        }

        System.out.println(Arrays.deepToString(b));
        double[][] c = C.correlate2D(b, a);
        System.out.println(Arrays.deepToString(c));

        System.out.println(Arrays.deepToString(Conv2D.convolve2DFull(c, a)));
    }
}
