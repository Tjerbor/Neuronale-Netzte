package function;


/**
 * This class models the signed disambiguation. function.
 */
public class Sgn extends Activation {

    @Override
    public double definition(double x) {
        return x > 0 ? 1 : -1;
    }

    @Override
    public double derivative(double x) {
        return 0;
    }

}
