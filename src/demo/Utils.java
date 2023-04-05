package demo;

public class Utils {

    /**
     * addiert die Biases nach der Matrix Multiplikation von
     * den Inputdaten und den Weights.
     *
     * @param inputs entspricht den Input * Weights.
     * @param bias   biases des layers.
     * @return gibt die inputs plus addierten bias zurück.
     */
    public static double[][] add_biases(double[][] inputs, double[] bias) {
        double[] out = new double[inputs[0].length];

        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                out[i] += inputs[i][j];
            }

        }

        return out;

    }

    /**
     * Bekommt die Weights zusammen mit den Biases übergeben, daher splittet sie Sie.
     * Diese Funktion gibt die Biases aus der bekommen 2D Matrix aus.
     *
     * @param a 2D Matrix mit biases und Weights obendrauf.
     * @return Biases
     */
    public static double[] split_for_biases(double[][] a) {
        return a[0];
    }

    /**
     * Bekommt die Weights zusammen mit den Biases übergeben, daher splittet sie Sie.
     * Diese Funktion die Biases von den Weights. Erwartet, dass die Biases auf Position Null
     * liegen.
     *
     * @param a 2D Matrix mit biases und Weights oben drauf.
     * @return Weights
     */
    public static double[][] split_for_weights(double[][] a) {
        double[][] weights = new double[a.length - 1][a[0].length];
        for (int i = 0; i < a.length - 1; i++) {
            weights[i] = a[i + 1];
        }
        return weights;
    }

    public static double[][] stack_array(double[] b, double[][] a) {


        //TODO needs to chceck for shape Execption
        double[][] result = new double[a.length + 1][b.length];
        result[0] = b;
        for (int i = 0; i < a.length; i++) {
            result[i + 1] = a[i];
        }

        return result;
    }

    /**
     * Fügt die Biases zu dem berechneten Ouput hinzu. (Inputs * Weights) + Biases.
     *
     * @param inputs berechneter Output.
     * @param bias   Werter der Biases
     * @return gibt die addierten Werte zurück.
     */
    public static double[] add_bias(double[] inputs, double[] bias) {
        for (int i = 0; i < inputs.length; i++) {
            inputs[i] += bias[i];
        }
        return inputs;
    }

    /**
     * Berechnet das Dot-Produkt für die gegeben Inputs und den Outputs.
     *
     * @param inputs
     * @param biases
     * @return
     */
    public static double[] dotProdukt_1D(double[][] inputs, double[] biases) {
        //adding the biases to the output.
        // addes the baises for the forward pass

        double[] output = new double[biases.length];
        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < biases.length; j++) {
                output[j] += inputs[i][j] * biases[j];
            }
        }

        return output;
    }

    /**
     * Transponiert eine gegeben Matrix.
     *
     * @param a inputs Matrix 2D.
     * @return
     */
    public double[][] tranpose(double[][] a) {
        //TODO

    }

    /**
     * Berechnet die Matrix Multiplikation.
     * Wird benutzt für dei Berechnung der Batch-Inputs und der Weights.
     *
     * @param input1 erwartet zuerst die Inputs.
     * @param input2 hier sollten die weights eingegeben werden.
     * @return gibt die Matrix multiplikation zurück.
     */
    public double[][] matmult2D(double[][] input1, double[][] input2) {
        //TODO
        return outputs;
    }

}
