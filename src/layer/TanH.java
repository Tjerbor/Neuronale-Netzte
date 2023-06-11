package layer;

/**
 * This class models the hyperbolic tangent function.
 */
public class TanH extends Activation {
    @Override
    public double definition(double x) {
        return Math.tanh(x);
    }

    @Override
    public double derivative(double x) {
        return 1 - (Math.pow(definition(x), 2));
    }


}
