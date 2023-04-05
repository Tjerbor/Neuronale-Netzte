package demo;

import src.Activations.*;


public class NeuronalesNetz_v2 {

    protected Layer[] Topologie;
    double schwellenwert;
    String function; //die gesetzte Funktion der

    int parameter_size; //Anzahl der enthaltenden Parameter des Models.

    public void create(int[] struktur) {
        //size -1, weil die erste Zahl die größe der Eingabe Daten entspricht.
        int model_size = struktur.length - 1; //länge der Topologie

        this.Topologie = new Layer[model_size];

        for (int i = 0; i < model_size; i++) {
            Topologie[i] = new FullyConnectedLayer(struktur[i], struktur[i + 1]);
            this.parameter_size += struktur[i] * struktur[i + 1] + struktur[i + 1];
        }

    }

    public double[][][] getWeights(int layer_pos) {
        double[][][] w = new double[Topologie.length][][];

        for (int i = 0; i < Topologie.length; i++) {
            double[][] weights = Topologie[i].weights;
            double[] biases = Topologie[i].biases;

            //fügt die Bias und weights zusammen zu einer Matrix.
            //Biases liegen auf dem Index 0.
            w[i] = Utils.stack_array(biases, weights);
        }

        return w;
    }

    public void setUnitType(int i, int j, String function, double theta) {
        //TODO
        this.schwellenwert = theta;
    }


    public void activation(double[][] inputs) {
        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; i++) {
                inputs[i][j] = Activations.useForwardFunktion(this.function, this.schwellenwert);

            }
        }
    }

    public void activation(double[] input) {
        for (int i = 0; i < input.length; i++) {
            input[i] = Activations.useForwardFunktion(this.function, this.schwellenwert);


        }
    }

    public void setWeights(double[][][] weights) {
        //needs to check if the shape is corrket(Layer Anzahl und gegeben Layer Anzahl.)

        for (int j = 0; j < Topologie.length; j++) {
            Topologie[j].weights = Utils.split_for_weights(weights[j]);
            Topologie[j].biases = Utils.split_for_biases(weights[j]);

        }

    }

    public double outputFunktion(double x) {
        if (x > this.schwellenwert) {
            return 1;
        }
        return 0;
    }


    /**
     * Berechnet die Ausgabe unsres Netzwerkes,
     * für ein einzlenes Sample.
     *
     * @param input Eingabe Daten.
     * @return gibt die Voraussage für ein einzelnes Sample aus,
     * ohne Loss-Funktion.
     */
    public double[] compute(double[] input) {
        //TODO

        double[] output = input;

        for (int i = 0; i < Topologie.length; i++) {
            output = Topologie[i].forward(output);
            // activation Funktion after each Layer.
            this.activation(output);

        }

        return output;
    }

    /**
     * Berechnet die Outputs für eine Menge (Batch) von Eingabe Daten.
     *
     * @param data Eingabe daten.
     * @return die Outputs der Layer, ohne Loss-Funktion
     */
    public double[][] computeAll(double[][] data) {
        double[][] outputs = data;
        for (int i = 0; i < Topologie.length; i++) {
            outputs = Topologie[i].forward(outputs);
            // activation Funktion after each Layer.
            this.activation(outputs);
        }


        return outputs;
    }
}
