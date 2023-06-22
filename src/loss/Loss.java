package loss;

/**
 * This interface models a loss function.
 */
public interface Loss {
    /**
     * This method calculates the loss function with the given input.
     */
    double forward(double[] actual, double[] expected);

    /**
     * This method calculates the loss function with the given inputs.
     */
    double forward(double[][] actual, double[][] expected);

    /**
     * This method calculates the inverse of the loss function with the given input.
     */
    double[] backward(double[] actual, double[] expected);

    /**
     * This method calculates the inverse of the loss function with the given inputs.
     */
    double[][] backward(double[][] actual, double[][] expected);
}
