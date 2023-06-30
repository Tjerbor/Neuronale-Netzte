package testFunctions;

import function.TanH;
import layer.ActivationLayer;
import layer.FCL;
import layer.FastLinearLayer;
import org.junit.jupiter.api.Test;
import utils.Matrix;
import utils.RandomUtils;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class TestFCL {

    @Test
    public void testFCL_Forward() {


        int inputSize = new Random().nextInt(40, 100);
        int neuron1 = new Random().nextInt(40, 300);
        double[] r = new double[inputSize];
        RandomUtils.genTypeWeights(2, r);

        FastLinearLayer fast1 = new FastLinearLayer(inputSize, neuron1);

        fast1.forward(new Matrix(r));

        FCL fcl1 = new FCL(inputSize, neuron1);
        fcl1.setUseBiases(false);
        ActivationLayer act1 = new ActivationLayer(new TanH());

        Matrix m = fast1.getWeights();
        fcl1.setWeights(m);

        fcl1.forward(new Matrix(r));

        act1.forward(fcl1.getOutput());
        Matrix m1 = act1.getOutput();

        Matrix m2 = fast1.getOutput();

        assertArrayEquals(m2.getData1D(), m1.getData1D());


    }

    @Test
    public void testFCL_Forward2() {


        int inputSize = new Random().nextInt(40, 100);
        int neuron1 = new Random().nextInt(40, 300);
        double[] r = new double[inputSize];
        RandomUtils.genTypeWeights(2, r);

        FastLinearLayer fast1 = new FastLinearLayer(inputSize, neuron1);

        fast1.forward(new Matrix(r));

        FCL fcl1 = new FCL(inputSize, neuron1);
        fcl1.setUseBiases(false);
        ActivationLayer act1 = new ActivationLayer(new TanH());

        fcl1.setNextLayer(act1);
        act1.setPreviousLayer(fcl1);

        Matrix m = fast1.getWeights();
        fcl1.setWeights(m);

        fcl1.forward(new Matrix(r));
        Matrix m1 = act1.getOutput();

        Matrix m2 = fast1.getOutput();

        assertArrayEquals(m2.getData1D(), m1.getData1D());


    }

    @Test
    public void testFCL_Backward() {


        int inputSize = new Random().nextInt(40, 100);
        int neuron1 = new Random().nextInt(40, 300);
        double[] r = new double[inputSize];
        RandomUtils.genTypeWeights(2, r);

        FastLinearLayer fast1 = new FastLinearLayer(inputSize, neuron1);

        fast1.forward(new Matrix(r));

        FCL fcl1 = new FCL(inputSize, neuron1);
        fcl1.setUseBiases(false);
        ActivationLayer act1 = new ActivationLayer(new TanH());

        fcl1.setNextLayer(act1);
        act1.setPreviousLayer(fcl1);

        Matrix m = fast1.getWeights();
        fcl1.setWeights(m);

        fcl1.forward(new Matrix(r));
        Matrix m1 = act1.getOutput();

        Matrix m2 = fast1.getOutput();

        assertArrayEquals(m2.getData1D(), m1.getData1D());


        double[] r2 = new double[neuron1];
        RandomUtils.genTypeWeights(2, r);
        fast1.backward(new Matrix(r2));
        m2 = fast1.getBackwardOutput();

        act1.backward(new Matrix(r2));

        m1 = fcl1.getBackwardOutput();


        assertArrayEquals(m2.getData1D(), m1.getData1D());

    }
}
