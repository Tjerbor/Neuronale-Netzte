import java.util.Arrays;
import java.util.stream.IntStream;

public class NeuronalesNetz {
    /**
     * This variable represents the identity function.
     */
    private static final String IDENTITY = "identity";
    /**
     * This variable represents a semi-linear function.
     */
    private static final String SEMI = "semi";
    /**
     * This variable represents a hard limiting threshold function.
     */
    private static final String HARD = "hard";
    /**
     * This variable represents a smoothly limiting threshold function.
     */
    private static final String SMOOTH = "smooth";
    /**
     * This variable represents the hyperbolic tangent function.
     */
    private static final String TANH = "tanh";
    /**
     * This array contains the number of units per layer, including bias units.
     * Its length corresponds to the number of layers of the neural network.
     */
    private int[] layers;
    /**
     * This array contains the activation functions of the individual units.
     */
    private String[][] functions;
    private double[][][] weights; //[Layer][Node][Edge] Includes BIAS nodes

    /**
     * This method initializes the neural network with the given number of units per layer.
     * It adds a bias unit to all layers except the output layer,
     * and sets the identity function as the activation function for all units.
     */
    public void create(int[] layers) {
        if (layers.length <= 1) {
            throw new IllegalArgumentException("The neural network must have more than one layer.");
        }

        this.layers = IntStream.range(0, layers.length).map(i -> layers[i] + (i < layers.length - 1 ? 1 : 0)).toArray();

        functions = new String[layers.length][Arrays.stream(layers).min().getAsInt()];

        for (int i = 0; i < layers.length; i++) {
            Arrays.fill(functions[i], IDENTITY);
        }
    }

    /**
     * Erste Klammer ist Knotenlayer, zweite Knoten i und dritte Kante j
     */
    public double[][][] getWeights() {
        return weights;
    }

    public void setWeights(double[][][] weights) {
        //TODO
    }

    public void setUnitType(int layer, int node, String function, double theta) {
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

    @Override
    public String toString() {
        return Arrays.toString(layers);
    }


}
