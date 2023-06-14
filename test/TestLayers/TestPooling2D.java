package TestLayers;

import layer.MaxPooling2DNew;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class TestPooling2D {


    @Test
    public void TestPoolingForward() {

        MaxPooling2DNew Pooling = new MaxPooling2DNew();

        double[][][] b = new double[1][6][6];

        int count = 0;
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                b[0][i][j] = count;
                count += 1;
            }
        }

        System.out.println(Arrays.deepToString(b));
        System.out.println(Arrays.deepToString(Pooling.forward(b)));


        double[][][] soll = new double[][][]{{{7, 9, 11,},
                {19, 21, 23},
                {31, 33, 35}}};

        assertArrayEquals(soll, Pooling.forward(b));

    }

    @Test
    public void TestPoolingBackward() {

        MaxPooling2DNew Pooling = new MaxPooling2DNew();

        double[][][] b = new double[1][6][6];

        int count = 0;
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                b[0][i][j] = count;
                count += 1;
            }
        }


        Arrays.deepToString(Pooling.forward(b));

        double[][][] bOut = Pooling.backward(b);


        double[][][] soll = new double[][][]
                {{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 7.0, 0.0, 9.0, 0.0, 11.0}, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                        {0.0, 19.0, 0.0, 21.0, 0.0, 23.0}, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 31.0, 0.0, 33.0, 0.0, 35.0}}};
        
        assertArrayEquals(soll, Pooling.backward(b));

    }


}
