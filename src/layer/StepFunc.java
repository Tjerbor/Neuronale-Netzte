package layer;

public class StepFunc extends Activation {

    double schwellenwert = 0;

    public StepFunc(double theta) {
        this.schwellenwert = theta;
    }

    public double def(double x) {
        if (x >= this.schwellenwert) {
            return 1;
        }

        return 0;
    }

    public double prime(double x) {
        return 0;
    }
}
