package function;

/**
 * This class models a piecewise constant function with two parts.
 */
public class StepFunc extends Activation {
    /**
     * This variable contains the threshold value.
     * The activation function is <code>1</code> if the given value is greater than or equal to this value.
     * It is <code>0</code> if the given value is less than this value.
     */
    private final double theta;

    /**
     * This constructor sets {@link StepFunc#theta}.
     */
    public StepFunc(double theta) {
        this.theta = theta;
    }

    @Override
    public double definition(double x) {
        return x >= theta ? 1 : 0;
    }

    @Override
    public double derivative(double x) {
        return 0;
    }
}
