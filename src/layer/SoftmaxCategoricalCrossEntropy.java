package layer;

import utils.Array_utils;

public class SoftmaxCategoricalCrossEntropy implements Loss {
    public static NewSoftmax softmax = new NewSoftmax();
    public static CategoricalCrossEntropy loss = new CategoricalCrossEntropy();

    @Override
    public double forward(double[] actual, double[] expected) {
        throw new UnsupportedOperationException();
    }

    @Override
    public double forward(double[][] actual, double[][] expected) {

        double[][] out = softmax.forward(actual);
        return loss.forward(out, expected);

    }

    @Override
    public double[] backward(double[] actual, double[] expected) {
        throw new UnsupportedOperationException();
    }

    @Override
    public double[][] backward(double[][] actual, double[][] expected) {

        //expects One-Hot-Encoded
        int batch_size = expected.length;

        double[][] out = new double[expected.length][expected[0].length];

        //either minus 0 or 1. class depending. //expect One-Hot-Encoded
        for (int i = 0; i < expected.length; i++) {
            for (int j = 0; j < expected[0].length; j++) {
                out[i][j] = actual[i][j] - expected[i][j];
            }
        }
        Array_utils.div_matrix_by_scalar(out, batch_size);
        return out;

    }
}
