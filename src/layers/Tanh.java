package layers;


public class Tanh extends Activation {

    public double def(double x) {
        return 1 - (2 / (Math.exp(2 * x) + 1));
    }

    public double prime(double x) {
        return 1 - (Math.pow(this.def(x), 2));
    }

}
