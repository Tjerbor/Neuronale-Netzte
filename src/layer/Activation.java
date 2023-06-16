package layer;

/**
 * This class is the superclass for all activation functions and models the identity function.
 */
public class Activation {

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


    public float definition(float x) {
        return x;
    }

    /**
     * This method evaluates the derivative of the activation function at the given point.
     */
    public float derivative(float x) {
        return 1;
    }
}
