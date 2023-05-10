package layer;

/**
 * This class models a semi-linear function.
 */
public class SemiLinear extends Activation {
    /**
     * This variable contains the upper limit that the function values do not exceed.
     */
    private final double upperLimit;
    /**
     * This variable contains the lower limit that the function values do not fall below.
     */
    private final double lowerLimit;

    public SemiLinear() {
        upperLimit = 1e+7;
        lowerLimit = 1e-7;
    }

    /**
     * This constructor sets the {@link SemiLinear#upperLimit} and the {@link SemiLinear#lowerLimit}.
     * It throws an exception if the upper limit is less than the lower limit.
     */
    public SemiLinear(double upperLimit, double lowerLimit) {
        if (upperLimit < lowerLimit) {
            throw new IllegalArgumentException("The upper limit must be greater than or equal to the lower limit.");
        }

        this.upperLimit = upperLimit;
        this.lowerLimit = lowerLimit;
    }

    @Override
    public double definition(double x) {
        return x > upperLimit ? upperLimit : Math.max(x, lowerLimit);
    }

    @Override
    public double derivative(double x) {
        return 0;
    }

    /**
     * This method returns the {@link SemiLinear#upperLimit} that the function values do not exceed.
     */
    public double getUpperLimit() {
        return upperLimit;
    }

    /**
     * This method returns the {@link SemiLinear#lowerLimit} that the function values do not fall below.
     */
    public double getLowerLimit() {
        return lowerLimit;
    }
}
