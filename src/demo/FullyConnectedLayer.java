package demo;

public class FullyConnectedLayer extends Layer {

    double[] biases; //biases of the Dense Layer
    double[][] weights; //weights of the layer
    double[][] outputs; //calculated outputs of the layer
    double inputs[][];
    double dinputs[][];


    public FullyConnectedLayer(int input_size, int output_size, double[] biases) {
        this.weights = new double[input_size][output_size];
        //this.biases = new double[output_size];

        if (biases.length != output_size) {
            //throws new Exception("Got wrong Values");
        } else {
            this.biases = biases;

        }


    }

    /**
     * Layer is created without any Values.
     *
     * @param input_size  Input size der daten oder des letzten Layers
     * @param output_size Eigene Anzahl von Neuronen.
     */
    public FullyConnectedLayer(int input_size, int output_size) {
        this.weights = new double[input_size][output_size];
        this.biases = new double[output_size];
    }

    public double[][] forward(double[][] inputs) {

        this.outputs = Utils.matmul2D(inputs, this.weights);
        this.outputs = Utils.add_biases(outputs, this.biases);
        return outputs;


    }

    public double[] forward(double[] input) {
        output = new double[input.length];
        output = Utils.dotProdukt_1D(Utils.transpose(this.weights), input);
        output = Utils.add_bias(output, this.biases);
        return output;
    }

    ;


}

