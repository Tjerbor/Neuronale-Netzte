package TestLayers;

import layer.Conv2D;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

public class TestConv2DC {


    @Test
    public void TestCv2D() {

        Conv2D c = new Conv2D();

        double[][][][] soll = new double[4][28][28][3];
        Arrays.fill(soll, 1);

        double[][][][] out = c.fordward(soll);

        System.out.println(Arrays.deepToString(soll));


    }
}
