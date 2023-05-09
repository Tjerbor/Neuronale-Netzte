package layer;

/**
 * The CustomActivation uses this function to calculate the activation of a node.
 */
public class activation_utils {
    /**
     * Prints out the supported Activation Functions.
     */
    public static void printSupportedActivations() {
        System.out.println("relu");
        System.out.println("tanh");
        System.out.println("sigmoid, logistisch, logi");
        System.out.println("semi, semi_linear, semi linear");
        System.out.println("id, identity, identity_function, identity function");
    }


    /**
     * used the forward dunction given by the node. is uesed by CustomActivation.
     *
     * @param type   name of the activation function.
     * @param x      value given by the node.
     * @param theata theata value. clip value or binary decision.
     * @return the computed value or identity.
     */
    public static double useForwardFunktion(String type, double x, double theata) {

        if (type.equals("relu")) {
            x = relu(x);
        } else if (type.equals("tanh")) {
            x = tahn(x);
        } else if (type.equals("sigmoid") || type.equals("logistisch") || type.equals("logi")) {
            x = sigmoid(x);

        } else if (type.equals("semi") || type.equals("semi_linear") || type.equals("semi linear")) {
            // is a clip for the given value for up and down.
            //otherwise returns the given value.
            if (x > theata) {
                x = theata;
            } else if (x < -theata) {
                x = -theata;
            }
        } else if (!type.equals("id")) {
            System.out.println("Activation Function is not Supported");
        }

        //means theta is set but,can be a zero value, which is not allowed.
        //Custom activation function sets all theata values to 0.
        //it is expected for 0 is relu used.
        // so this check is needed.
        if (!(type.equals("semi") || type.equals("semi_linear") || type.equals("semi linear")) && theata != 0) {
            if (x > theata) {
                x = 1;
            } else {
                x = 0;
            }
        }

        return x;
    }

    /**
     * used the forward dunction given by the node. is uesed by CustomActivation.
     *
     * @param type name of the activation function.
     * @param x    value given by the node.
     * @return the computed value or identity.
     */
    public static double useForwardFunktion(String type, double x) {
        String[] split = type.split(",");

        if (split[0].equals("relu")) {
            x = relu(x);
        } else if (split[0].equals("tanh")) {
            x = tahn(x);
        } else if (type.equals("sigmoid") || type.equals("logistisch") || type.equals("logi")) {
            x = sigmoid(x);
        } else if (split[0].equals("one")) {
            x = 1;
        } else if (!type.equals("id")) {
            System.out.println("Activation Function is not Supported");
        }
        //semi linear activation Function is not allowed,
        //because it needs a theata so set upper and lower Boundaries.

        /*
        Checks if calculted value x must be gated with theta value.
        If true, then x get normalized to 1 or 0 depending on theta.
        */
        return x;
    }


    public static double useBackwardFunktion(String type, double x, double theata) {

        if (type.equals("relu")) {
            x = relu_prime(x);
        } else if (type.equals("tanh")) {
            x = tahn_prime(x);
        } else if (type.equals("sigmoid") || type.equals("logistisch") || type.equals("logi")) {
            x = sigmoid_prime(x);

        } else if (type.equals("semi") || type.equals("semi_linear") || type.equals("semi linear")) {
            // is a clip for the given value for up and down.
            //otherwise returns the given value.
            if (x > theata) {
                x = theata;
            } else if (x < -theata) {
                x = -theata;
            }
        } else if (!type.equals("id") | !type.equals("identity") | !type.equals("identity_function") | !type.equals("identity function")) {
            System.out.println("Activation Function is not Supported");
        }

        //means theta is set but,can be a zero value, which is not allowed.
        //Custom activation function sets all theata values to 0.
        //it is expected for 0 is relu used.
        // so this check is needed.
        if (!(type.equals("semi") || type.equals("semi_linear") || type.equals("semi linear")) && theata != 0) {
            if (x > theata) {
                x = 1;
            } else {
                x = 0;
            }
        }

        return x;
    }

    /**
     * used the forward dunction given by the node. is uesed by CustomActivation.
     *
     * @param type name of the activation function.
     * @param x    value given by the node.
     * @return the computed value or identity.
     */
    public static double useBackwardFunktion(String type, double x) {
        String[] split = type.split(",");

        if (split[0].equals("relu")) {
            x = relu_prime(x);
        } else if (split[0].equals("tanh")) {
            x = tahn_prime(x);
        } else if (split[0].equals("sigmoid")) {
            x = sigmoid_prime(x);
        } else if (split[0].equals("one")) {
            x = 1;
        } else if (!type.equals("id") | !type.equals("identity") | !type.equals("identity_function") | !type.equals("identity function")) {
            System.out.println("Activation Function is not Supported");
        }
        //semi linear activation Function is not allowed,
        //because it needs a theata so set upper and lower Boundaries.

        /*
        Checks if calculted value x must be gated with theta value.
        If true, then x get normalized to 1 or 0 depending on theta.
        */
        return 1;
    }

    public static double relu(double x) {

        if (x > 0) {
            return x;
        } else {
            return 0;
        }

    }

    //die Backward rectified linear unit wichtig für die Backpropagation.
    //Ableitung der relu Funktion.
    public static double relu_prime(double x) {
        if (x > 0) {
            return 1;
        } else {
            return 0;
        }
    }


    //Berechnet die Sigmoid Funktion für gegebenen Input.
    // andere Name logistic_regression.
    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));

    }

    //Die Ableitung der Sigmoid Funktion.
    public static double sigmoid_prime(double x) {
        double s = Math.log(x);
        return s * (1 - s);

    }

    //Die Hyperbolic Tangent Funktion
    public static double tahn(double x) {
        return 1 - (2 / (Math.exp(2 * x) + 1));
    }

    public static double tahn_prime(double x) {
        return 2 * Math.log(2 * x) - 1;
    }

}
