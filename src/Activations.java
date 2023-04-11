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

        if (split[0].equals("relu")) {
            x = this.relu(x);
        } else if (split[0].equals("tahn")) {
            x = this.tahn(x);
        } else if (split[0].equals("sigmoid")) {
            x = this.sigmoid(x);
        } else if (split[0].equals("one")) {
            x = 1;
        }

        /*
        Checks if calculted value x must be gated with theta value.
        If true, then x get normalized to 1 or 0 depending on theta.
        */
        return split.length == 2 ? (x >= Double.valueOf(split[1]) ? 1 : 0) : x;
    }

    private double relu(double x) {

        if (x > 0) {
            return x;
        } else {
            return 0;
        }

    }

    //die Backward rectified linear unit wichtig für die Backpropagation.
    //Ableitung der relu Funktion.
    private double relu_prime(double x) {
        if (x > 0) {
            return 1;
        } else {
            return 0;
        }
    }


    //Berechnet die Sigmoid Funktion für gegebenen Input.
    // andere Name logistic_regression.
    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));

    }

    //Die Ableitung der Sigmoid Funktion.
    private double sigmoid_prime(double x) {
        double s = Math.log(x);
        return s * (1 - s);

    }

    //Die Hyperbolic Tangent Funktion
    private double tahn(double x) {
        return 1 - (2 / (Math.exp(2 * x) + 1));
    }

    private double tahn_prime(double x) {
        return 2 * Math.log(2 * x) - 1;
    }


}
