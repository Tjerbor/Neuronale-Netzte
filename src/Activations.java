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
        String[] split = type.split(",");

        if(split.length == 2){ //Function has Theta Value
            return useForwardFunktion(split[0],Double.parseDouble(split[1]),x);
        }


        if (type.equals("relu")) {
            x = this.relu(x);
        } else if (type.equals("tahn")) {
            x = this.tahn(x);
        } else if (type.equals("sigmoid")) {
            x = this.sigmoid(x);
        } else if (type.equals("one")) {
            x = 1;
        }

        return x;
    }

    private double useForwardFunktion(String type, double Theta, double x) {
        //TODO
        return -1;
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
    // andere Name logistic_regression.
    public double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));

    }

    //Die Ableitung der Sigmoid Funktion.
    public double sigmoid_prime(double x) {
        double s = Math.log(x);
        return s * (1 - s);

    }

    //Die Hyperbolic Tangent Funktion
    public double tahn(double x) {
        return 1 - (2 / (Math.exp(2 * x) + 1));
    }

    public double tahn_prime(double x) {
        return 2 * Math.log(2 * x) - 1;
    }


}
