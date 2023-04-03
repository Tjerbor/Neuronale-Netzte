public class Activations {

    //computes the activation function of a given value.
    //prime funktionen sind der Backward-Pass (Die Ableitungen der gegeben Funktionen)
    //prime Funktionen für die Backpropagation.

    //die Forward rectified linear unit.

    // Alle Aktivierungs-Funktion nehmen nur ein einzelnen output wert an.

    /**
     * Entescheide welche Aktivierungs-Funktion genutzt wird.
     *
     * @param type enthält den namen der Funktion
     * @param x    is der wert des weights
     * @return returned den berechneten wert des outputs.
     */
    public double useForwardFunktion(String type, double x) {


        if (type.equals("relu")) {
            x = this.relu(x);
        } else if (type.equals("tahn")) {
            x = this.tahn(x);
        } else if (type.equals("sigmoid")) {
            x = this.sigmoid(x);
        }

        return x;
    }


    /**
     * verwendet die identitäts-Funktion als Aktivierungs-Funktion.
     *
     * @param x wert des outputs
     * @return returend den output
     */
    public double useForwardFunktion(double x) {
        return x;
    }


    public double relu(double x) {

        if (x > 0) {
            return x;
        } else {
            return 0;
        }

    }

    //die Backward rectified linear unit wichtig für die Backpropagation.
    //Ableitung der relu Funktion.
    public double relu_prime(double x) {
        if (x > 0) {
            return 1;
        } else {
            return 0;
        }
    }


    //Berechnet die Sigmoid Funktion für gegebenen Input.
    public double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));

    }

    //Die Ableitung der Sigmoid Funktion.
    public double sigmoid_prime(double x) {
        double s = this.sigmoid(x);
        return s * (1 - s);

    }

    //Die Hyperbolic Tangent FunKtion
    public double tahn(double x) {
        return 1 - (2 / (Math.exp(2 * x) + 1));
    }

    public double tahn_prime(double x) {
        return 1 - Math.pow(this.tahn(x), 2);
    }


}
