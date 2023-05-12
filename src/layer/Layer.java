package layer;

/**
 * This interface models one layer of the neural network.
 *
 * @see layer.FullyConnectedLayer
 * @see layer.Activation
 */
public interface Layer {
    /**
     * initialize the Momentum weights and Biases with zeros.
     */
    aktivateMomentum();

    double[][] getWeights();

    void setWeights(double[][] weights);

    double[][] getMomentumWeights();

    double[][] getDeltaWeights();

    double[] getBiases();

    double[] getDeltaBiases();

    double[] getMomentumBiases();

    /**
     * This method returns the number of parameters of the layer.
     */
    int parameters();

    /**
     * This method calculates a forward pass with the given input.
     */
    double[] forward(double[] input);

    /**
     * This method calculates a forward pass with each of the given inputs.
     */
    double[][] forward(double[][] inputs);

    /**
     * This method calculates a backward pass with the given input.
     */
    double[] backward(double[] input);

    /**
     * This method calculates a backward pass with the given input and the given learning rate.
     */
    double[] backward(double[] input, double learningRate);

    /**
     * This method calculates a backward pass with each of the given inputs.
     */
    double[][] backward(double[][] inputs);

    /**
     * This method calculates a backward pass with each of the given inputs and the given learning rate.
     */
    double[][] backward(double[][] inputs, double learningRate);
}
