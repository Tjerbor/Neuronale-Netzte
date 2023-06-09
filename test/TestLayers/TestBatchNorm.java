package TestLayers;

import layer.BatchNorm1D;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

public class TestBatchNorm {


    @Test
    public void Simple1D() {


        double[][] input = new double[][]{{0.92227751, 0.99016412, 0.7974192},
                {0.89949949, 0.35887433, 0.85769244}};

        double[][] soll = new double[][]{{0.99999615, 0.99999999,
                -0.99999945}, {-0.99999615, -0.99999999, 0.99999945}};


        //Shape is 2, 3
        // Batch Size Samples.
        BatchNorm1D B1D = new BatchNorm1D(3);

        B1D.setMomentum(0.9);
        B1D.setUseMomentum(true);
        B1D.setEpsilon(1e-9);
        double[][] test_output = B1D.forward(input);

        System.out.println(Arrays.deepToString(test_output));

        double[][] out = B1D.backwardNew(soll);
        System.out.println(Arrays.deepToString(out));

        double[][] sollBack = new double[][]{{
                6.76916937e-04, 3.17982849e-08, -3.65355306e-05},
                {-6.76916937e-04, -3.17982836e-08, 3.65355304e-05}};

    }


}