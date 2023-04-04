package demo;

public class Utils {

    /**
     * @param inputs
     * @param bias
     * @return gibt die inputs plus addierten bias zurück.
     */
    public static double[] add_biases(double[][] inputs, double[] bias) {
        double[] out = new double[inputs[0].length];

        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                out[i] += inputs[i][j];
            }

        }

        return out;

    }
    public static double[] add_bias(double[] inputs, double[] bias){
        for (int i = 0; i < inputs.length; i++) {
            inputs[i] += bias[i];
        }
        return bias;
    }

    public tranpose(double[][] a){
        //TODO
    }
}
    public static double[] dotProdukt_1D(double[][] inputs, double[] biases){
        //adding the biases to the output.
        // addes the baises for the forward pass

        double[] output = new double[biases.length];
        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < biases.length; j++) {
                output[j] += inputs[i][j] * biases[j];
            }}

        return output;

    /**
     * @param input1 erwartet zuerst die Inputs.
     * @param input2 hier sollten die weights eingegeben werden.
     * @return gibt die Matrix multiplikation zurück.
     */
    public double[][] matmult2D(double[][] input1, double[][] input2) {
        //TODO
        return outputs;
    }

}
