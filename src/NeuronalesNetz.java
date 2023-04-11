import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
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
     * This variable represents the bias function, which always returns <code>1</code>.
     */
    private static final String ONE = "one";
    /**
     * This array contains the number of units per layer, including bias units.
     * Its length corresponds to the number of layers of the neural network.
     */
    private int[] layers;
    /**
     * This array contains the activation functions of the individual units, including bias nodes.
     * The value for the unit <code>j</code> in layer <code>i</code> can be found at the index <code>[i][j]</code>.
     */
    private String[][] functions;
    /**
     * This array contains the weights that define the connections between the units, including bias units.
     * The value for the connection between unit <code>j</code> in layer <code>i</code>
     * and unit <code>k</code> in the next layer can be found at index <code>[i][j][k]</code>.
     */
    private double[][][] weights;

    /**
     * This method attempts to read the given CSV file and return the values it contains.
     * It throws an exception if the file does not exist or an I/O error occurs.
     */
    private static List<String[]> read(String path) throws IOException {
        try (BufferedReader in = new BufferedReader(new FileReader(path))) {
            String line;

            List<String[]> list = new ArrayList<>();

            while ((line = in.readLine()) != null) {
                String[] values = line.split(";");

                list.add(values);
            }

            return list;
        }
    }

    /**
     * This method initializes the neural network with the given number of units per layer.
     * It adds a bias unit to all layers except the output layer,
     * sets the identity function as the activation function for all units,
     * and initializes the weights with random values between <code>-1</code> and <code>1</code>.
     */
    public void create(int[] layers) {
        if (layers.length <= 1) {
            throw new IllegalArgumentException("The neural network must have more than one layer.");
        }

        this.layers = IntStream.range(0, layers.length).map(i -> {
            if (layers[i] < 1) {
                throw new IllegalArgumentException("Each layer must have at least one unit.");
            }

            return layers[i] + (i < layers.length - 1 ? 1 : 0);
        }).toArray();

        functions = new String[layers.length][];

        for (int i = 0; i < functions.length; i++) {
            functions[i] = new String[this.layers[i]];

            Arrays.fill(functions[i], IDENTITY);

            if (i != functions.length - 1) {
                functions[i][functions[i].length - 1] = ONE; //Sets BIAS functions
            }

        }

        this.fillWeights();
    }

    /**
     * This method reads the given CSV file and initializes the neural network with the values it contains.
     * It adds a bias unit to all layers except the output layer,
     * and sets the identity function as the activation function for all units.
     */
    public void create(String path) throws IOException {
        List<String[]> list = read(path);

        if (!list.get(0)[0].equals("layers")) {
            throw new IllegalArgumentException();
        }

        String[] layers = list.get(0);

        this.layers = IntStream.range(1, layers.length).map(i -> {
            int number = Integer.parseInt(layers[i]);

            if (number < 1) {
                throw new IllegalArgumentException("Each layer must have at least one unit.");
            }

            return number;
        }).toArray();

        // TODO
    }

    /**
     * Erste Klammer ist Knotenlayer, zweite Knoten i und dritte Kante j
     */
    public double[][][] getWeights() {
        return weights;
    }

    public void setWeights(double[][][] weights) {
        this.weights = weights;
    }

    public void setUnitType(int layer, int node, String function, double theta) {
        functions[layer][node] = function + ',' + Double.toString(theta);
    }

    public void setUnitType(int layer, int node, String function) {
        functions[layer][node] = function;
    }

    public double[] compute(double[] input) throws IllegalArgumentException {
        if (input.length != layers[0] - 1) { //Need to subtract BIAS-neuron
            throw new IllegalArgumentException("Input amount does not equal neuron amount in first layer");
        }

        double[] Input;
        double[] Output = new double[input.length + 1];
        for (int i = 0; i < input.length; i++) {
            Output[i] = input[i];
        }

        for (int i = 0; i < layers.length; i++) {
            Input = Output;
            Output = calculateLayer(i, Input);
        }

        return Output;
    }

    public double[][] computeAll(double[][] data) {
        double[][] Output = new double[data.length][];
        for (int i = 0; i < Output.length; i++) {
            Output[i] = this.compute(data[i]);
        }
        return Output;
    }

    @Override
    public String toString() {
        return Arrays.toString(layers);
    }

    private void fillWeights() {
        Random r = new Random();
        this.weights = new double[layers.length - 1][][];
        for (int layer = 0; layer < layers.length - 1; layer++) {
            weights[layer] = new double[layers[layer]][];
            for (int i = 0; i < weights[layer].length; i++) {
                weights[layer][i] = new double[layers[layer + 1]];
                for (int j = 0; j < weights[layer][i].length; j++) {
                    weights[layer][i][j] = r.nextDouble(-1.0, 1 + Double.MIN_VALUE);
                }
            }
        }
    }

    private double[] calculateLayer(int layer, double[] input) {
        Activations act = new Activations();
        double[] output;
        if (layer + 1 < layers.length) {
            output = new double[layers[layer + 1]];
            for (int node = 0; node < layers[layer]; node++) {
                for (int edge = 0; edge < weights[layer][node].length; edge++) {
                    output[edge] += act.useForwardFunktion(functions[layer][node], input[node]) * weights[layer][node][edge];
                }
            }
        } else {
            output = input;
            for (int node = 0; node < output.length; node++) {
                output[node] = act.useForwardFunktion(functions[layer][node], input[node]);
            }
        }
        return output;
    }


}
