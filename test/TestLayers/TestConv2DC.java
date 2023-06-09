package TestLayers;

import layer.Conv2D;
import layer.MaxPooling2D;
import org.junit.jupiter.api.Test;
import utils.Array_utils;

import java.util.Arrays;

import static utils.Array_utils.getShape;

public class TestConv2DC {


    @Test
    public void TestCv2D() {

        Conv2D c = new Conv2D(32, 3);
        MaxPooling2D poll = new MaxPooling2D();

        double[][][][] soll = new double[4][28][28][3];
        soll = Array_utils.fill(soll, 1);

        double[][][][] out = c.forward(soll);
        System.out.println(Arrays.toString(getShape(out)));
        soll = poll.forward(out);

        System.out.println(Arrays.deepToString(soll));
        System.out.println(Arrays.toString(getShape(soll)));


    }
}
