package loss;

/**
 * This class models the mean squared error function.
 */
public class MSE implements Loss {
    public static final String EXCEPTION = "The arrays must have the same length.";

    @Override
    public double forward(double[] actual, double[] expected) {
        if (actual.length != expected.length) {
            throw new IllegalArgumentException(EXCEPTION);
        }

        double sum = 0;

        for (int i = 0; i < actual.length; i++) {
            sum += Math.pow(actual[i] - expected[i], 2);
        }

        return sum / actual.length;
    }

    @Override
    public double forward(double[][] actual, double[][] expected) {
        if (actual.length != expected.length) {
            throw new IllegalArgumentException(EXCEPTION);
        }

        double sum = 0;

        for (int i = 0; i < actual.length; i++) {
            sum += forward(actual[i], expected[i]);
        }

        return sum / actual.length;
    }

    @Override
    public double[] backward(double[] actual, double[] expected) {
        if (actual.length != expected.length) {
            throw new IllegalArgumentException(EXCEPTION);
        }

        double[] result = new double[actual.length];

        for (int i = 0; i < actual.length; i++) {
            result[i] = 2 * (actual[i] - expected[i]) / actual.length;
        }

        return result;
    }

    @Override
    public double[][] backward(double[][] actual, double[][] expected) {
        if (actual.length != expected.length) {
            throw new IllegalArgumentException(EXCEPTION);
        }

        double[][] result = new double[actual.length][];

        for (int i = 0; i < actual.length; i++) {
            result[i] = backward(actual[i], expected[i]);
        }

        return result;
    }
}
