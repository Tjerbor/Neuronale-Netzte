package demo;


public class NeuronalesNetz_v2 {

    protected Layer[] Topologie;

    public void create(int[] struktur) {
        //size -1, weil die erste Zahl die größe der Eingabe Daten entspricht.
        int model_size = struktur.length - 1; //länge der Topologie

        this.Topologie = new Layer[model_size];


        for (int i = 0; i < model_size; i++) {
            Topologie[i] = new FullyConnectedLayer(struktur[i], struktur[i + 1]);

        }

    }

    public double[][][] getWeights(int layer_pos) {
        //TODO
        return null;
    }

    public void setUnitType(int i, int j, String function, double theta) {
        //TODO
    }

    public void setWeights(double[][][] weights) {
        //TODO
    }

    public double[] compute(double[] input) {
        //TODO

        double[] output = input;

        for (int i = 0; i < Topologie.length; i++) {
            output = Topologie[i].forward(output);
            // activation Funktion after each Layer.

        }
        

        return output;
    }

    public double[][] computeAll(double[][] data) {
        double[][] output = data;
        for (int i = 0; i < Topologie.length; i++) {
            output = Topologie[i].forward(output);
            // activation Funktion after each Layer.

        }
        return null;
    }
}
