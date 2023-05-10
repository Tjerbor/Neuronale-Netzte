package layer;

/**
 * This class models a sigmoid function, namely the logistic function.
 */
public class Sigmoid extends Activation {
    @Override
    public double definition(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    @Override
    public double derivative(double x) {
        double y = definition(x);

        return y * (1 - y);
    }
}
