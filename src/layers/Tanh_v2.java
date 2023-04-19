package layers;

public class Tanh_v2 extends Activation {

    public double def(double x) {
        return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
    }

    public double prime(double x) {
        return 1 - (Math.pow(this.def(x), 2));
    }
}
