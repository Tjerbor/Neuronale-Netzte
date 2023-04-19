package layers;

public class Sigmoid extends Activation{
    public double def (double x){
        return 1 / (1 + Math.exp(-x));

    }
    public double prime(double x){
        double s = this.def(x);
        return s * (1-s);

    }

}
