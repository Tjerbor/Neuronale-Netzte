package layer;

public class ReLu extends Activation {

    public ReLu() {
        name = "relu";
    }

    public double def(double x) {

        if (x > 0) {
            return x;
        } else {
            return 0;
        }

    }

    public double prime(double x) {
        if (x > 0) {
            return 1;
        } else {
            return 0;
        }
    }


}
