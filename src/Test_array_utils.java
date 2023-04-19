import utils.Array_utils;

import java.util.Arrays;

public class Test_array_utils {

    public static void main(String[] args) {

        double[] a = Array_utils.linspace_wo_endpoint(-1, 1, 20, 4);
        System.out.println(Arrays.toString(a));

        double dec = 0;
        try {
            dec = Array_utils.roundDec(-0.29999999999999993, 8);
        } catch (Exception e) {
            System.out.println(e);
        }

        System.out.println(dec);

        double[][] w = Array_utils.getLinspaceWeights_wo_endpoint(10, 20, -1, 1, 4);


        Array_utils.printMatrix(w);


    }
}
