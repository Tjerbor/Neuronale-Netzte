package layer;

import utils.Array_utils;

public class SoftmaxCategoricalCrossEntropy extends Losses {

    public static Softmax softmax = new Softmax();
    public static CategoricalCrossEntropy loss = new CategoricalCrossEntropy();

    public double forward(double[][] inputs, double[][] y_true) {

        double[][] out = softmax.forward(inputs);
        return loss.forward(out, y_true);

    }


    public double[][] backward(double[][] dvalues, double[][] y_true) {

        //expects One-Hot-Encoded
        int batch_size = y_true.length;

        double[][] out = new double[y_true.length][y_true[0].length];

        //either minus 0 or 1. class depending. //expect One-Hot-Encoded
        for (int i = 0; i < y_true.length; i++) {
            for (int j = 0; j < y_true[0].length; j++) {
                out[i][j] = dvalues[i][j] - y_true[i][j];
            }
        }
        Array_utils.div_matrix_by_scalar(out, batch_size);
        return out;

    }


}