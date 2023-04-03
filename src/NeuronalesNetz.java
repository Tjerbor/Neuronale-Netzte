import java.util.Arrays;

public class NeuronalesNetz {
    private int[] layers; //Length of layer, includes BIAS nodes
    private String[][] nodeFunctions; //[Layer][Node]
    private double[][][] weights; //[Layer][Node][Edge] Includes BIAS nodes


    public void create(int[] struktur) {
        //TODO
    }

    /**
     * Erste Klammer ist Knotenlayer, zweite Knoten i und dritte Kante j
     */
    public double[][][] getWeights() {
        return weights;
    }

    public void setUnitType(int layer, int node, String function, double theta) {
        //TODO
    }

    public void setWeights(double[][][] weights) {
        //TODO
    }

    public double[] compute(double[] input) throws IllegalArgumentException {
        if (input.length != weights[0].length - 1) { //Need to subtract BIAS-neuron
            throw new IllegalArgumentException("Input amount does not equal neuron amount in first layer");
        }
        double[] Iteration = input;
        double[] newIteration = new double[layers[1]];
        newIteration[newIteration.length - 1] = 1; //BIAS neuron always outputs 1

        for (int layer = 0; layer < weights.length - 1; layer++) {
            for (int node = 0; node < weights[layer].length; node++) {
                for (int edge = 0; edge < weights[layer][node].length; edge++) {
                   // newIteration[edge] += weights[layer][node][edge] *
                }
            }
        }

        return Arrays.copyOfRange(newIteration, 0, (newIteration.length - 1));
    }

    public double[][] computeAll(double[][] data) {
        //TODO
        return null;
    }

    public NeuronalesNetz read_CSV(String filePath) {
        //TODO
        return null;
    }

    public String toString() {
        //TODO
        return null;
    }


}
