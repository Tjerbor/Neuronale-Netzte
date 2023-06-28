package TestLayers;

import layer.DropoutLayer;
import org.junit.jupiter.api.Test;
import utils.RandomUtils;

import java.util.Arrays;

public class TestDropout {


    @Test
    public void testDropout() {

        DropoutLayer dropoutLayer = new DropoutLayer(0.5);
        
        double[] a = new double[10];
        RandomUtils.genTypeWeights(2, a);
        dropoutLayer.forward(a);
        dropoutLayer.backward(a);
        double[] ist1D = dropoutLayer.getOutput().getData1D();
        double[] soll1D = dropoutLayer.getBackwardOutput().getData1D();

        assert Arrays.equals(ist1D, soll1D);

    }
}
