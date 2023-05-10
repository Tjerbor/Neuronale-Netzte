package layer;

/**
 * This class models the rectified linear unit function.
 */
public class ReLu extends Activation {
    @Override
    public double definition(double x) {
        return x > 0 ? x : 0;
    }

    @Override
    public double derivative(double x) {
        return x > 0 ? 1 : 0;
    }
}
