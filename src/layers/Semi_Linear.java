package layers;

public class Semi_Linear extends Activation{

    double limit = 1e-7;
    public double def(double x){

        if (x > this.limit){
            return limit;
        } else if (x < -limit) {
            return -limit;
        }
        return x;
    }

    public double prime(double x){
        return 0;
    }
}
