package TestNewModel;

import extraLayer.FullyConnectedLayer;
import main.NeuralNetwork;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

public class TestModel {

    @Test
    void testModelAdd() {


        NeuralNetwork nn = new NeuralNetwork();

        nn.add(new FullyConnectedLayer(7, 1));
        nn.add(new FullyConnectedLayer(3, 4));
        nn.add(new FullyConnectedLayer(17, 5));


        System.out.println(Arrays.toString(nn.getLayers()));

    }

    @Test
    void testModelCompute() {


        NeuralNetwork nn = new NeuralNetwork();

        nn.add(new FullyConnectedLayer(10, 3));
        nn.add(new FullyConnectedLayer(3, 4));
        nn.add(new FullyConnectedLayer(4, 5));


        double[] a = new double[10];
        Arrays.fill(a, 1);

        System.out.println(Arrays.toString(nn.compute(a)));


    }

    @Test
    void testModelBackward() {


        NeuralNetwork nn = new NeuralNetwork();

        nn.add(new FullyConnectedLayer(10, 3));
        nn.add(new FullyConnectedLayer(3, 4));
        nn.add(new FullyConnectedLayer(4, 5));


        double[] a = new double[10];
        Arrays.fill(a, 1);

        a = nn.compute(a);
        System.out.println(Arrays.toString((a)));
        nn.computeBackward(a);
        System.out.println(Arrays.toString(a));


    }
}
