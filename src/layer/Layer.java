package layer;

/**
 * This interface models one layer of the neural network.
 *
 * @see layer.FullyConnectedLayer
 * @see layer.Activation
 */
public interface Layer {
    double[][] getWeights();

    void setWeights(double[][] weights);

    double[] forward(double[] input);

    double[][] forward(double[][] inputs);

    double[] backward(double[] input);

    double[] backward(double[] input, double learningRate);

    double[][] backward(double[][] inputs);

    double[][] backward(double[][] inputs, double learningRate);
}
