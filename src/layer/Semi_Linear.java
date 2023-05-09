package layer;

public class Semi_Linear extends Activation {

    double limit = 1e-7;

    public Semi_Linear() {
        this.name = "semi_linear";
    }

    public double def(double x) {

        if (x > this.limit) {
            return limit;
        } else if (x < -limit) {
            return -limit;
        }
        return x;
    }

    public double prime(double x) {
        return 0;
    }
}
