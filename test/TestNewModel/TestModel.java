package TestNewModel;

import main.FullyConnectedLayerNew;
import main.NN_New;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

public class TestModel {

    @Test
    void testModelAdd() {


        NN_New nn = new NN_New();

        nn.add(new FullyConnectedLayerNew(7, 1));
        nn.add(new FullyConnectedLayerNew(3, 4));
        nn.add(new FullyConnectedLayerNew(17, 5));


        System.out.println(Arrays.toString(nn.getLayers()));

    }

    @Test
    void testModelCompute() {


        NN_New nn = new NN_New();

        nn.add(new FullyConnectedLayerNew(10, 3));
        nn.add(new FullyConnectedLayerNew(3, 4));
        nn.add(new FullyConnectedLayerNew(4, 5));


        double[] a = new double[10];
        Arrays.fill(a, 1);

        System.out.println(Arrays.toString(nn.compute(a)));


    }

    @Test
    void testModelBackward() {


        NN_New nn = new NN_New();

        nn.add(new FullyConnectedLayerNew(10, 3));
        nn.add(new FullyConnectedLayerNew(3, 4));
        nn.add(new FullyConnectedLayerNew(4, 5));


        double[] a = new double[10];
        Arrays.fill(a, 1);

        a = nn.compute(a);
        System.out.println(Arrays.toString((a)));
        nn.computeBackward(a);
        System.out.println(Arrays.toString(a));


    }
}
