public class Activations {

    //computes the activation function of a given value.
    //prime funktionen sind der Backward-Pass (Die Ableitungen der gegeben Funktionen)
    //prime Funktionen für die Backpropagation.

    //die Forward rectified linear unit.
    public double relu(double x){

        if (x > 0) {
            return x;
        }else{return 0;}

    }

    //die Backward rectified linear unit. wichtig für die Backpropagation.
    //Ableitung der
    public double relu_prime(double x){
        if (x > 0) {
            return 1;
        }else{return 0;}
    }


    //Berechnet die Sigmoid Funktion für gegebenen Input.
    public double sigmoid(double x){
        return 1 / (1 + Math.exp(-x));

    }
    //Die Ableitung der Sigmoid Funktion.
    public double sigmoid_prime(double x){
        double s = this.sigmoid(x);
        return s * (1-s);

    }

    //Die hyberbolic
    public double tahn(double x){
        return 1 -  (2 / (Math.exp(2 * x) + 1));
    }

    public double tahn_prime(double x){
        return  1 - Math.pow(this.tahn(x), 2);
    }






}
