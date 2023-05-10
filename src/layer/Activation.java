package layer;

/**
 * This class is the superclass for all activation functions and models the identity function.
 */
public class Activation implements Layer {
    double[] input;
    double[][] inputs;

    /**
     * This method multiplies the corresponding elements of the given arrays.
     * It throws an exception if the arrays do not have the same length.
     */
    private static double[] multiply(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new ArithmeticException("The arrays must have the same length.");
        }

        double[] result = new double[a.length];

        for (int i = 0; i < result.length; i++) {
            result[i] = a[i] * b[i];
        }

        return result;
    }

    /**
     * This method multiplies the corresponding elements of the given arrays.
     * It throws an exception if the arrays do not have the same length.
     */
    private static double[][] multiply(double[][] a, double[][] b) {
        if (a.length != b.length) {
            throw new ArithmeticException("The arrays must have the same length.");
        }

        double[][] result = new double[a.length][a[0].length];

        for (int i = 0; i < result.length; i++) {
            result[i] = multiply(a[i], b[i]);
        }

        return result;
    }

    /**
     * This method evaluates the activation function at the given point.
     */
    public double definition(double x) {
        return x;
    }

    /**
     * This method evaluates the derivative of the activation function at the given point.
     */
    public double derivative(double x) {
        return 1;
    }

    /**
     * This method throws an {@link UnsupportedOperationException}.
     */
    @Override
    public double[][] getWeights() {
        throw new UnsupportedOperationException();
    }

    /**
     * This method throws an {@link UnsupportedOperationException}.
     */
    @Override
    public void setWeights(double[][] weights) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int parameters() {
        return 0;
    }

    @Override
    public double[] forward(double[] input) {
        double[] output = new double[input.length];

        for (int i = 0; i < input.length; i++) {
            output[i] = definition(input[i]);
        }

        return output;
    }

    @Override
    public double[][] forward(double[][] inputs) {
        double[][] output = new double[inputs.length][inputs[0].length];

        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                output[i][j] = definition(inputs[i][j]);
            }
        }

        return output;
    }

    @Override
    public double[] backward(double[] input) {
        double[] output = new double[input.length];

        for (int i = 0; i < input.length; i++) {
            output[i] = derivative(input[i]);
        }

        return multiply(input, output);
    }

    @Override
    public double[] backward(double[] input, double learningRate) {
        return backward(input);
    }

    @Override
    public double[][] backward(double[][] inputs) {
        double[][] outputs = new double[inputs.length][inputs[0].length];

        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                outputs[i][j] = derivative(inputs[i][j]);
            }

            outputs = multiply(inputs, outputs);
        }

        return outputs;
    }

    @Override
    public double[][] backward(double[][] inputs, double learningRate) {
        return backward(inputs);
    }
}
